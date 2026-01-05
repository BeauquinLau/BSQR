#### --- High-Dimensional Scalability & Efficiency Comparison --- ####
# Objective: Demonstrate the "Curse of Dimensionality" in PETEL vs HMC Efficiency in BSQR
# Settings: n=300, p=50 (High Dimension)
# Metric: ESS (Effective Sample Size), ESS/Second, R-hat
#
# Hardware Note: 
# Benchmarks were performed on a MacBook Air (Apple M4 chip). 
# Absolute timings and ESS/sec will vary by hardware; relative performance trends should hold.
rm(list = ls()); gc()
set.seed(2025)

library(MASS)
library(quantreg)
library(cmdstanr)
library(posterior)
library(tidyverse)
library(parallel)

#### --- 1. Tang & Yang (PETEL) Faithful Implementation --- ####

er_petel <- function(theta, X, y, tau) {
  resid <- y - X %*% theta
  rho <- ifelse(resid < 0, resid * (tau - 1), resid * tau)
  return(mean(rho))
}

gr_petel <- function(theta, xi, yi, tau) {
  resid <- yi - sum(xi * theta)
  k <- ifelse(resid == 0, 0, ifelse(resid > 0, -tau, 1 - tau))
  return(k * xi)
}

# The bottleneck function: Optimization in High Dimension
lel_petel <- function(theta, X, y, tau) {
  n <- nrow(X); d <- ncol(X)
  Xg <- matrix(0, nrow = n, ncol = d)
  
  # This double loop O(n*d) becomes heavy when d=50
  for (i in 1:n) Xg[i, ] <- gr_petel(theta, X[i,], y[i], tau)
  
  lambda <- rep(0, d)
  iend <- 1
  
  # Newton-Raphson in d-dimensions
  # FIX: Increased max iterations to 50 for high-dim convergence stability
  for (k in 1:50) { 
    if (iend == 1) {
      arg <- as.vector(Xg %*% lambda)
      if(max(arg) > 50) return(-Inf)
      probs <- exp(arg)
      
      # Hessian construction O(d^2 * n) - Expensive!
      H <- t(Xg) %*% (probs * Xg) / n 
      G <- t(Xg) %*% probs / n
      
      step <- tryCatch(solve(H, G), error = function(e) NULL)
      if(is.null(step)) return(-Inf)
      
      lambda <- lambda - step
      if (sqrt(sum(step^2)) <= 1e-6) iend <- 0
    }
  }
  
  # Check convergence (optional strictly, but affects accuracy)
  if(iend == 1) return(-Inf) # Treat non-convergence as failure
  
  log_sum_exp_term <- sum(exp(Xg %*% lambda))
  log_el <- sum(Xg %*% lambda) - n * log(log_sum_exp_term)
  if(is.nan(log_el)) return(-Inf)
  return(log_el)
}

# Random Walk Metropolis (Suffers in High Dim)
run_petel_mcmc <- function(X, y, tau, n_iter=2000, burnin=1000) {
  n <- nrow(X); d <- ncol(X)
  alpha <- 2 * sqrt(n)
  
  # Init via RQ (Fast frequentist start)
  init_beta <- tryCatch(coef(rq(y ~ X-1, tau=tau)), error=function(e) rep(0, d))
  if(any(is.na(init_beta))) init_beta <- rep(0, d)
  
  curr_beta <- init_beta
  
  # Start timing from here to include initialization overhead (Fairness)
  start_time <- Sys.time()
  
  curr_lel <- lel_petel(curr_beta, X, y, tau)
  curr_loss <- er_petel(curr_beta, X, y, tau)
  curr_log_prior <- -0.5 * sum(curr_beta^2)
  
  if(curr_lel == -Inf) return(list(success=FALSE))
  
  chain <- matrix(0, nrow=n_iter, ncol=d)
  accept <- 0
  
  # Adaptive step size (simplistic) to try and help PETEL
  step_scale <- 0.05 
  
  for(t in 1:n_iter) {
    # Random Walk proposal in 50-dimensions is very inefficient
    prop_beta <- curr_beta + rnorm(d, 0, step_scale)
    
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
  end_time <- Sys.time()
  
  return(list(
    success=TRUE, 
    chain=chain[(burnin+1):n_iter, ], 
    time=as.numeric(difftime(end_time, start_time, units="secs")),
    accept_rate = accept/n_iter
  ))
}

#### --- 2. BSQR Setup --- ####
# Only using Gaussian kernel for High-Dim efficiency demo (it's the fastest/smoothest)
stan_model_gaussian <- cmdstanr::cmdstan_model("bsqr_gaussian_Z_robust.stan", quiet=TRUE)

#### --- 3. Simulation Config --- ####
n_sim <- 300       # Need larger N to support P=50
p_dim <- 50        # High Dimension!
n_reps <- 20       # Fewer reps because high-dim is slow
tau <- 0.5
# Sparse signal: First 5 are 1, rest are 0
beta_true <- c(rep(1, 5), rep(0, p_dim - 5)) 

# Helper to calculate ESS and Rhat
calc_diagnostics <- function(chain, time_sec, method_name) {
  # chain dim: (draws, p)
  
  # Use 'posterior' package for robust ESS/Rhat
  draws_obj <- as_draws_matrix(chain)
  
  # Calculate min ESS (bulk) across all parameters
  ess_vals <- summarise_draws(draws_obj, default_convergence_measures())
  
  # Handle potential NAs in ESS calculation (if chain is constant or too short)
  min_ess <- min(ess_vals$ess_bulk, na.rm=TRUE)
  if(is.infinite(min_ess)) min_ess <- 0
  
  max_rhat <- max(ess_vals$rhat, na.rm=TRUE)
  if(is.infinite(max_rhat)) max_rhat <- NA
  
  # ESS per Second (The ultimate efficiency metric)
  ess_per_sec <- min_ess / time_sec
  
  # MSE
  est <- colMeans(chain)
  mse <- mean((est - beta_true)^2)
  
  return(data.frame(
    Method = method_name,
    Time_Sec = time_sec,
    Min_ESS = min_ess,
    Max_Rhat = max_rhat,
    ESS_per_Sec = ess_per_sec,
    MSE = mse,
    Dimensions = p_dim
  ))
}

#### --- 4. Main Simulation Loop --- ####
all_results <- list()

cat("\nStarting High-Dim Efficiency Test (n =", n_sim, ", p =", p_dim, ")\n")
pb <- txtProgressBar(min = 0, max = n_reps, style = 3)

for(i in 1:n_reps) {
  set.seed(2025 + i)
  
  # Data Gen
  X <- matrix(rnorm(n_sim * p_dim), ncol=p_dim)
  err <- rnorm(n_sim, 0, 1) 
  y <- as.vector(X %*% beta_true + err)
  
  # --- 1. Run PETEL ---
  # Note: This will be slow due to O(d^2) optimization inside MCMC
  petel_out <- run_petel_mcmc(X, y, tau, n_iter=2000, burnin=1000)
  
  if(petel_out$success) {
    res_petel <- calc_diagnostics(petel_out$chain, petel_out$time, "PETEL")
  } else {
    res_petel <- data.frame(Method="PETEL", Time_Sec=NA, Min_ESS=0, Max_Rhat=NA, ESS_per_Sec=0, MSE=NA, Dimensions=p_dim)
  }
  all_results[[length(all_results)+1]] <- res_petel
  
  # --- 2. Run BSQR (HMC) ---
  h_val <- 1.06 * sd(y) * n_sim^(-1/5)
  
  s_data <- list(N_train_obs=n_sim, K=p_dim, X_train=X, y_train=y, 
                 tau=tau, h=h_val, beta_location=rep(0, p_dim), beta_scale=rep(10, p_dim),
                 gamma_shape=0.01, theta_prior_rate_val=0.01, upper_bound_for_theta=20, epsilon_theta=1e-4, 
                 Z_rel_tol=1e-4, K_ASYMPTOTIC_SWITCH_STD_DEVS=5, USE_Z_ASYMPTOTIC_APPROX=1)
  
  init_fun <- function() list(beta_raw=rnorm(p_dim, 0, 0.1), theta=1)
  
  start_t <- Sys.time()
  # Run Stan (1 chain is enough for speed comparison)
  fit <- stan_model_gaussian$sample(data=s_data, chains=1, iter_warmup=1000, iter_sampling=1000, 
                                    refresh=0, show_messages=FALSE, init=init_fun)
  end_t <- Sys.time()
  bsqr_time <- as.numeric(difftime(end_t, start_t, units="secs"))
  
  # Correctly identifying the parameter name based on Gaussian stan file
  chain_bsqr <- fit$draws("beta_params", format="matrix")
  res_bsqr <- calc_diagnostics(chain_bsqr, bsqr_time, "BSQR (HMC)")
  
  all_results[[length(all_results)+1]] <- res_bsqr
  
  setTxtProgressBar(pb, i)
}
close(pb)

#### --- 5. Summary Table --- ####
final_df <- bind_rows(all_results)

summary_table <- final_df %>%
  group_by(Method) %>%
  summarise(
    Avg_Time = mean(Time_Sec, na.rm=TRUE),
    Avg_Min_ESS = mean(Min_ESS, na.rm=TRUE),
    Avg_ESS_per_Sec = mean(ESS_per_Sec, na.rm=TRUE),
    Avg_Rhat = mean(Max_Rhat, na.rm=TRUE),
    Avg_MSE = mean(MSE, na.rm=TRUE)
  )

print(summary_table)
write.csv(summary_table, "Table_Scalability_HighDim.csv")