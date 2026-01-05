#### --- Finite Sample & Symmetric Extreme Quantile Stress Test --- ####
# Objective: 
# 1. Assess efficiency at the median (Tau=0.5)
# 2. Assess robustness at extreme tails (Tau=0.05, 0.95)
# Comparison: BSQR (4 Kernels) vs PETEL (Tang & Yang, 2022)
# Setting: Extreme sparsity (n=30, p=6) with heteroscedastic errors

# --- A. Clean Environment & Set Seed ---
rm(list = ls()) 
gc()
set.seed(2025) 

# --- B. Load Packages ---
# Check if cmdstanr is installed, if not, try to install or stop
if (!requireNamespace("cmdstanr", quietly = TRUE)) {
  stop("Package 'cmdstanr' is required. Please install it first.")
}

library(MASS)
library(quantreg)
library(cmdstanr)
library(posterior)
library(tidyverse)
library(parallel)

# --- C. Check CmdStan Path ---
# This step is crucial to fix the "No CmdStan config" error
# It forces cmdstanr to verify its installation path before running models
check_cmdstan_toolchain(fix = TRUE, quiet = TRUE)

if (is.null(cmdstan_path())) {
  stop("CmdStan path not found. Please run install_cmdstan() or set_cmdstan_path().")
}

#### --- 1. PETEL Implementation (Tang & Yang, 2022) --- ####
# Faithful implementation of the projected empirical likelihood method

# Check Loss Function
er_petel <- function(theta, X, y, tau) {
  resid <- y - X %*% theta
  rho <- ifelse(resid < 0, resid * (tau - 1), resid * tau)
  return(mean(rho))
}

# Estimating Equations (Gradient of Check Loss)
gr_petel <- function(theta, xi, yi, tau) {
  resid <- yi - sum(xi * theta)
  k <- ifelse(resid == 0, 0, ifelse(resid > 0, -tau, 1 - tau))
  return(k * xi)
}

# Log-Empirical Likelihood Calculation
lel_petel <- function(theta, X, y, tau) {
  n <- nrow(X); d <- ncol(X)
  Xg <- matrix(0, nrow = n, ncol = d)
  for (i in 1:n) Xg[i, ] <- gr_petel(theta, X[i,], y[i], tau)
  
  lambda <- rep(0, d)
  iend <- 1
  
  # Newton-Raphson Optimization for Lagrange Multipliers
  for (k in 1:20) { 
    if (iend == 1) {
      arg <- as.vector(Xg %*% lambda)
      if(max(arg) > 50) return(-Inf) 
      
      probs <- exp(arg)
      H <- t(Xg) %*% (probs * Xg) / n
      G <- t(Xg) %*% probs / n
      
      step <- tryCatch(solve(H, G), error = function(e) NULL)
      if(is.null(step)) return(-Inf) 
      
      lambda <- lambda - step
      if (sqrt(sum(step^2)) <= 1e-6) iend <- 0
    }
  }
  
  log_sum_exp_term <- sum(exp(Xg %*% lambda))
  log_el <- sum(Xg %*% lambda) - n * log(log_sum_exp_term)
  
  if(is.nan(log_el)) return(-Inf)
  return(log_el)
}

# MCMC Sampler for PETEL (Random Walk Metropolis)
run_petel_mcmc <- function(X, y, tau, n_iter=3000, burnin=1000) {
  n <- nrow(X); d <- ncol(X)
  alpha <- 2 * sqrt(n)
  
  # Initialization via standard quantile regression
  init_beta <- tryCatch(coef(rq(y ~ X-1, tau=tau)), error=function(e) rep(0, d))
  if(any(is.na(init_beta))) init_beta <- rep(0, d)
  
  curr_beta <- init_beta
  curr_lel <- lel_petel(curr_beta, X, y, tau)
  curr_loss <- er_petel(curr_beta, X, y, tau)
  curr_log_prior <- -0.5 * sum(curr_beta^2) 
  
  if(curr_lel == -Inf) return(list(success=FALSE))
  
  chain <- matrix(0, nrow=n_iter, ncol=d)
  accept <- 0
  
  for(t in 1:n_iter) {
    prop_beta <- curr_beta + rnorm(d, 0, 0.1) 
    prop_lel <- lel_petel(prop_beta, X, y, tau)
    
    if(prop_lel == -Inf) {
      log_ratio <- -Inf
    } else {
      prop_loss <- er_petel(prop_beta, X, y, tau)
      prop_log_prior <- -0.5 * sum(prop_beta^2)
      log_ratio <- (prop_lel - curr_lel) - alpha * (prop_loss - curr_loss) + (prop_log_prior - curr_log_prior)
    }
    
    if(log(runif(1)) < log_ratio) {
      curr_beta <- prop_beta; curr_lel <- prop_lel; curr_loss <- prop_loss; curr_log_prior <- prop_log_prior
      accept <- accept + 1
    }
    chain[t,] <- curr_beta
  }
  return(list(success=TRUE, chain=chain[(burnin+1):n_iter, ]))
}

#### --- 2. BSQR Setup --- ####
stan_files <- list(
  gaussian = "bsqr_gaussian_Z_robust.stan",
  uniform = "bsqr_uniform_Z_robust.stan",
  epanechnikov = "bsqr_epanechnikov_Z_robust.stan",
  triangular = "bsqr_triangular_Z_robust.stan"
)

stan_models <- list()
for(k_name in names(stan_files)) {
  if(file.exists(stan_files[[k_name]])) {
    stan_models[[k_name]] <- cmdstanr::cmdstan_model(stan_files[[k_name]], quiet=TRUE)
  }
}

#### --- 3. Simulation Configuration --- ####
n_sim <- 30
p_dim <- 6
n_reps <- 100

# Quantiles to evaluate
tau_list <- c(0.05, 0.5, 0.95)

# True parameters (Sparse signal)
beta_true <- c(1, 1, rep(0, p_dim - 2))

# Helper function to calculate metrics
calc_metrics <- function(chain, beta_true, method_name, kernel_name, tau_val) {
  est <- colMeans(chain)
  bias_sq <- mean((est - beta_true)^2) 
  
  ci <- apply(chain, 2, quantile, probs=c(0.025, 0.975))
  width <- mean(ci[2,] - ci[1,])
  cov <- mean((beta_true >= ci[1,]) & (beta_true <= ci[2,]))
  
  alpha_sig <- 0.05
  scores <- (ci[2,] - ci[1,]) + 
    (2/alpha_sig) * (ci[1,] - beta_true) * (beta_true < ci[1,]) + 
    (2/alpha_sig) * (beta_true - ci[2,]) * (beta_true > ci[2,])
  is_score <- mean(scores)
  
  est_named <- setNames(est, paste0("B", 1:length(est)))
  
  res <- data.frame(
    Tau = tau_val, 
    Method = method_name, 
    Kernel = kernel_name,
    MSE_Rep = bias_sq, 
    Width = width, 
    Cov = cov, 
    IS_Score = is_score, 
    Failure = 0
  )
  return(cbind(res, t(est_named)))
}

#### --- 4. Main Simulation Loop --- ####
all_results <- list()

cat("\nStarting Symmetric Stress Test (n =", n_sim, ")\n")

for(tau_val in tau_list) {
  cat("\nProcessing Tau =", tau_val, "...\n")
  pb <- txtProgressBar(min = 0, max = n_reps, style = 3)
  
  for(i in 1:n_reps) {
    # Ensure distinct seed for each rep and tau
    set.seed(888 + i + which(tau_list == tau_val)*1000)
    
    # Data Generation: Heteroscedastic Errors
    X <- matrix(rnorm(n_sim * p_dim), ncol=p_dim)
    err <- rnorm(n_sim, 0, 1 + 0.5*abs(X[,1])) 
    y <- as.vector(X %*% beta_true + err)
    
    # --- Run PETEL ---
    petel_out <- run_petel_mcmc(X, y, tau_val)
    
    if(petel_out$success) {
      res_petel <- calc_metrics(petel_out$chain, beta_true, "PETEL", "N/A", tau_val)
    } else {
      # Record Failure
      res_petel <- data.frame(Tau=tau_val, Method="PETEL", Kernel="N/A", MSE_Rep=NA, Width=NA, Cov=NA, IS_Score=NA, Failure=1)
      res_petel[paste0("B", 1:p_dim)] <- NA
    }
    all_results[[length(all_results)+1]] <- res_petel
    
    # --- Run BSQR ---
    # Bandwidth selection rule
    h_val <- 1.06 * sd(y) * n_sim^(-1/5)
    
    for(k_name in names(stan_models)) {
      s_data <- list(N_train_obs=n_sim, K=p_dim, X_train=X, y_train=y, 
                     tau=tau_val,
                     h=h_val, beta_location=rep(0, p_dim), beta_scale=rep(10, p_dim),
                     gamma_shape=0.01, upper_bound_for_theta=20, epsilon_theta=1e-4, 
                     Z_rel_tol=1e-4, K_ASYMPTOTIC_SWITCH_STD_DEVS=5, USE_Z_ASYMPTOTIC_APPROX=1)
      
      if(k_name %in% c("gaussian", "uniform")) {
        s_data$theta_prior_rate_val <- 0.01
      } else {
        s_data$gamma_rate <- 0.01 
        if(k_name == "epanechnikov") {
          s_data$Z_outer_integration_lower_bound <- -150
          s_data$Z_outer_integration_upper_bound <- 150
        }
      }
      
      init_fun <- function() list(beta_raw=rnorm(p_dim, 0, 0.1), theta=1)
      
      # Try running Stan model
      fit <- tryCatch(
        stan_models[[k_name]]$sample(data=s_data, chains=1, iter_warmup=500, iter_sampling=1000, 
                                     refresh=0, show_messages=FALSE, init=init_fun, adapt_delta = 0.9, max_treedepth = 12),
        error = function(e) NULL
      )
      
      if(!is.null(fit)) {
        var_name <- if(k_name == "gaussian") "beta_params" else "beta"
        chain_bsqr <- fit$draws(var_name, format="matrix")
        res_bsqr <- calc_metrics(chain_bsqr, beta_true, "BSQR", str_to_title(k_name), tau_val)
      } else {
        res_bsqr <- data.frame(Tau=tau_val, Method="BSQR", Kernel=str_to_title(k_name), MSE_Rep=NA, Width=NA, Cov=NA, IS_Score=NA, Failure=1)
        res_bsqr[paste0("B", 1:p_dim)] <- NA
      }
      all_results[[length(all_results)+1]] <- res_bsqr
    }
    
    setTxtProgressBar(pb, i)
  }
  close(pb)
}

#### --- 5. Summary Table Calculation --- ####
final_df <- bind_rows(all_results)

# Group by Tau, Method, and Kernel
summary_table <- final_df %>%
  group_by(Tau, Method, Kernel) %>%
  summarise(
    Failure_Rate = mean(Failure) * 100,
    MSE = mean(MSE_Rep, na.rm=TRUE),
    
    # Calculate Squared Bias across all coefficients
    Bias_Sq = mean(c(
      (mean(B1, na.rm=T) - beta_true[1])^2,
      (mean(B2, na.rm=T) - beta_true[2])^2,
      (mean(B3, na.rm=T) - beta_true[3])^2,
      (mean(B4, na.rm=T) - beta_true[4])^2,
      (mean(B5, na.rm=T) - beta_true[5])^2
    )),
    Var = MSE - Bias_Sq,
    IS_Score = mean(IS_Score, na.rm=TRUE),
    Cov = mean(Cov, na.rm=TRUE) * 100,
    .groups = "drop"
  ) %>%
  # Custom Sorting Logic:
  # 1. Method: PETEL first, then BSQR
  # 2. Kernel: N/A -> Gaussian -> Uniform -> Epanechnikov -> Triangular
  mutate(
    Method_Sort = factor(Method, levels = c("PETEL", "BSQR")),
    Kernel_Sort = factor(Kernel, levels = c("N/A", "Gaussian", "Uniform", "Epanechnikov", "Triangular"))
  ) %>%
  arrange(Tau, Method_Sort, Kernel_Sort) %>%
  # Remove sorting helper columns
  select(Tau, Method, Kernel, Failure_Rate, MSE, Bias_Sq, Var, IS_Score, Cov)

# Output results
print(summary_table, n=100)
write.csv(summary_table, "Table_Stress_Test_Sparsity.csv")