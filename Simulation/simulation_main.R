#### --- 0. Clear Environment and Set Seed --- ####
rm(list = ls())
gc()
set.seed(123)

# --- Set Working Directory ---
if (interactive() && requireNamespace("rstudioapi", quietly = TRUE)) {
  current_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
  if (nzchar(current_dir)) {
    setwd(current_dir)
    cat("Working directory set to:", current_dir, "\n")
  } else {
    cat("Could not determine script directory. Please set working directory manually.\n")
  }
} else {
  sourced_file_path <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NULL)
  if (!is.null(sourced_file_path) && nzchar(dirname(sourced_file_path))) {
    current_dir <- dirname(sourced_file_path)
    setwd(current_dir)
    cat("Working directory set to (based on sourced script):", current_dir, "\n")
  } else {
    cat("Not running in RStudio or rstudioapi not available. Could not determine script directory.\n")
    cat("Current working directory:", getwd(), "\n")
  }
}

#### --- 0.1 Check and install 'remotes' package if missing --- ####
if (!require("remotes")) {
  install.packages("remotes")
}

remotes::install_github("stan-dev/cmdstanr")
options(download.file.method = "libcurl")
library(cmdstanr)

#### --- 1. Environment and Packages --- ####
required_packages <- c("MASS", "quantreg", "cmdstanr", "rstan", "tidyverse",
                       "matrixStats", "knitr", "posterior", "stringr", "brms", "parallel", "caret",
                       "future", "future.apply", "progressr")

install_if_missing <- function(pkg_name) {
  if (!requireNamespace(pkg_name, quietly = TRUE)) {
    cat(paste0("Installing package: ", pkg_name, "\n"))
    install.packages(pkg_name, dependencies = TRUE)
  }
  library(pkg_name, character.only = TRUE)
}
invisible(sapply(required_packages, install_if_missing))

# --- CmdStan Setup ---
if (is.null(cmdstanr::cmdstan_path()) || is.null(tryCatch(cmdstanr::cmdstan_version(error_on_NA = FALSE), error = function(e) NULL))) {
  cat("CmdStan not found or path not set correctly. Attempting to install...\n")
  tryCatch({
    cmdstanr::install_cmdstan(cores = max(1, parallel::detectCores() - 1), overwrite = FALSE, quiet = FALSE)
    cmdstanr::check_cmdstan_toolchain(fix = TRUE, quiet = TRUE)
  }, error = function(e) {
    message("CmdStan installation failed or was skipped. Please install CmdStan manually or ensure cmdstan_path() is set.")
    stop(e)
  })
  if (is.null(cmdstanr::cmdstan_path()) || is.null(tryCatch(cmdstanr::cmdstan_version(error_on_NA = FALSE), error = function(e) NULL))) {
    stop("CmdStan installation was not successful. Please check CmdStan setup.")
  }
}
cat("CmdStan found at:", cmdstanr::cmdstan_path(), "\n")
cat("CmdStan version:", cmdstanr::cmdstan_version(), "\n")
cmdstanr::check_cmdstan_toolchain(fix = TRUE, quiet = TRUE)


#### --- 2. Simulation Parameters --- ####
M <- 200 # Number of MC replications per setting
n_train <- 200
n_test <- 1000
rho_x_val <- 0.5
K_ASYMPTOTIC_SWITCH_STD_DEVS_val <- 5.0

USE_Z_ASYMPTOTIC_APPROX_val <- 1 # For kernels that support it (Gaussian, Epan, Triangular)

beta_scenarios <- list(
  sparse = list(name = "Sparse_p20", p = 20, beta_true = c(3, 1.5, 0, 0, 2, rep(0, 15))),
  dense = list(name = "Dense_p8", p = 8, beta_true = rep(0.85, 8))
)

taus_sim <- c(0.25, 0.5, 0.75)

error_distributions <- list(
  list(type = "normal", params = list(mean = 0, sd = 1), name_suffix = "norm_0_1"),
  list(type = "t", params = list(df = 3), name_suffix = "t3_unscaled"),
  list(type = "mixture_normal",
       params = list(p1 = 0.2, m1 = 0, s1 = sqrt(3), p2 = 0.8, m2 = 0, s2 = 2.0),
       name_suffix = "mix_norm_v3v4"),
  list(type = "hetero_normal",
       params = list(gamma0 = -0.25, gamma1 = 0.5),
       name_suffix = "hetero_norm_x1")
)

# Shared Prior Settings
beta_prior_mean_val <- 0
beta_prior_var_val <- 1000
theta_prior_a_r <- 0.01 # Corresponds to gamma_shape
theta_prior_b_r <- 0.01 # Corresponds to gamma_rate / theta_prior_rate_val
upper_bound_theta_r_val <- 25.0
epsilon_theta_val <- 1e-6
Z_rel_tol_val <- 1e-6
Z_outer_integration_lower_bound_val <- -150
Z_outer_integration_upper_bound_val <- 150

# brms-specific Priors
sigma_ald_prior_scale_brms <- 2.5
beta_prior_sd_brms <- base::sqrt(beta_prior_var_val)

### CV PARAMETERS ###
USE_CV_FOR_H <- 1
CV_N_FOLDS <- 5
CV_H_GRID_FACTORS <- c(0.5, 0.75, 1.0, 1.5, 2.0)
CV_MCMC_PARAMS <- list(
  iter_warmup = 300,
  iter_sampling = 500,
  chains = 2,
  parallel_chains = 2,
  thin = 1,
  refresh = 0,
  show_messages = FALSE,
  adapt_delta = 0.9,
  max_treedepth = 10
)

USE_TEST_MCMC_SETTINGS <- FALSE

if (USE_TEST_MCMC_SETTINGS) {
  n_total_iter_mcmc <- 2000; n_warmup_mcmc <- 1000
  n_chains_mcmc <- 2; n_thin_mcmc <- 1
  cat("USING TEST MCMC SETTINGS\n")
} else {
  n_total_iter_mcmc <- 4000; n_warmup_mcmc <- 2000
  n_chains_mcmc <- 2; n_thin_mcmc <- 1
  cat("USING ORIGINAL MCMC SETTINGS\n")
}
n_sampling_iter_mcmc <- n_total_iter_mcmc - n_warmup_mcmc

### PARALLELIZATION PARAMETERS ###
N_WORKERS_OUTER <- floor(parallel::detectCores() / 2)
CORES_PER_INNER_SAMPLER <- min(n_chains_mcmc, max(1, floor(parallel::detectCores() / N_WORKERS_OUTER)))
if(N_WORKERS_OUTER == 0) N_WORKERS_OUTER <- 1
if(CORES_PER_INNER_SAMPLER == 0) CORES_PER_INNER_SAMPLER <- 1

cat(paste("Outer loop parallelization: Using", N_WORKERS_OUTER, "workers.\n"))
cat(paste("Inner MCMC samplers (brms, Stan) will use up to", CORES_PER_INNER_SAMPLER, "core(s) for parallel chains within each worker.\n"))
cat(paste("Total MCMC chains for main models:", n_chains_mcmc, "\n"))


#### --- 3. Helper Functions (R side) --- ####
generate_data <- function(n, p, beta_true_mean_effect, rho_x, error_type = "normal", error_params = list()) {
  Sigma_X <- base::matrix(0, nrow = p, ncol = p)
  for (j in 1:p) { for (k in 1:p) { Sigma_X[j, k] <- rho_x^(base::abs(j - k)) } }
  X <- MASS::mvrnorm(n, mu = base::rep(0, p), Sigma = Sigma_X)
  
  error_component <- switch(
    error_type,
    "normal" = { stats::rnorm(n, mean = error_params$mean, sd = error_params$sd) },
    "t" = { stats::rt(n, df = error_params$df) },
    "mixture_normal" = {
      component_choice <- stats::rbinom(n, 1, error_params$p1)
      draws1 <- stats::rnorm(n, error_params$m1, error_params$s1)
      draws2 <- stats::rnorm(n, error_params$m2, error_params$s2)
      base::ifelse(component_choice == 1, draws1, draws2)
    },
    "hetero_normal" = {
      if (p < 1) base::stop("Heteroscedastic model requires at least one covariate.")
      sigma_i <- base::exp(error_params$gamma0 + error_params$gamma1 * X[, 1])
      stats::rnorm(n, mean = 0, sd = sigma_i)
    },
    base::stop(base::paste("Unsupported error_type:", error_type))
  )
  
  y <- X %*% beta_true_mean_effect + error_component
  return(base::list(X = X, y = base::as.vector(y), u_true_error_component = error_component, Sigma_X = Sigma_X))
}

calculate_h_silverman <- function(residuals_vec, num_obs) {
  if (base::length(residuals_vec) < 2) return(0.1)
  sigma_u_hat <- stats::sd(residuals_vec, na.rm = TRUE)
  if (base::is.na(sigma_u_hat) || sigma_u_hat == 0) {
    sigma_u_hat <- stats::mad(residuals_vec, na.rm = TRUE, constant = 1.4826)
  }
  if (base::is.na(sigma_u_hat) || sigma_u_hat == 0) return(0.1)
  h_calc <- 1.06 * sigma_u_hat * num_obs^(-1/5)
  return(base::max(h_calc, 5e-3))
}

rho_tau_loss <- function(u, tau_level) {
  base::ifelse(u >= 0, tau_level * u, (tau_level - 1) * u)
}

provide_inits <- function(p_dims, current_epsilon_theta, current_upper_theta, prior_theta_shape, prior_theta_rate) {
  init_theta_candidate <- prior_theta_shape / prior_theta_rate
  init_theta_candidate <- base::max(current_epsilon_theta + 1e-4, base::min(current_upper_theta - 1e-4, init_theta_candidate))
  init_theta <- base::max(current_epsilon_theta + 1e-5, init_theta_candidate)
  init_theta <- base::min(current_upper_theta - 1e-5, init_theta)
  if (base::is.na(init_theta) || base::is.infinite(init_theta) || init_theta >= (current_upper_theta - 1e-6) || init_theta <= (current_epsilon_theta + 1e-6)) {
    init_theta <- (current_epsilon_theta + current_upper_theta) / 2.0
    init_theta <- base::max(current_epsilon_theta + 1e-5, base::min(current_upper_theta - 1e-5, init_theta))
  }
  base::list(
    beta_raw = stats::rnorm(p_dims, 0, 0.1),
    theta = init_theta
  )
}

# Unified CV function
perform_cv_for_h <- function(X_train, y_train, tau, h_grid, n_folds,
                             stan_model_obj_for_cv, kernel_name,
                             cv_mcmc_params_worker, base_stan_data_list, seed, log_prefix_cv) {
  base::cat(base::paste0(log_prefix_cv, " starting perform_cv_for_h for ", kernel_name, " kernel. seed: ", seed, " @@@\n"))
  
  if (base::is.null(stan_model_obj_for_cv) || !base::inherits(stan_model_obj_for_cv, "CmdStanModel")) {
    base::cat(base::paste0(log_prefix_cv, " - ERROR: Stan model for CV (", kernel_name, ") is NULL or invalid.\n"))
    return(h_grid[base::ceiling(base::length(h_grid)/2)])
  }
  
  base::set.seed(seed)
  folds <- caret::createFolds(y_train, k = n_folds, list = TRUE, returnTrain = FALSE)
  avg_check_losses <- base::numeric(base::length(h_grid))
  base::names(avg_check_losses) <- base::as.character(h_grid)
  p_current <- base::ncol(X_train)
  
  for (i in base::seq_along(h_grid)) {
    h_candidate <- h_grid[i]
    fold_losses <- base::numeric(n_folds)
    
    for (k in 1:n_folds) {
      val_indices <- folds[[k]]
      train_indices <- base::setdiff(1:base::length(y_train), val_indices)
      
      X_cv_train <- X_train[train_indices, , drop = FALSE]
      y_cv_train <- y_train[train_indices]
      X_cv_val <- X_train[val_indices, , drop = FALSE]
      y_cv_val <- y_train[val_indices]
      
      # Use the base stan data list and update it
      cv_stan_data <- base_stan_data_list
      cv_stan_data$N_train_obs <- base::nrow(X_cv_train)
      cv_stan_data$X_train <- X_cv_train
      cv_stan_data$y_train <- y_cv_train
      cv_stan_data$h <- h_candidate
      
      # Determine the correct parameter names for init function
      theta_rate_param_name <- if (kernel_name %in% c("epanechnikov", "triangular")) "gamma_rate" else "theta_prior_rate_val"
      init_fun_cv <- function() provide_inits(p_current, cv_stan_data$epsilon_theta, cv_stan_data$upper_bound_for_theta, cv_stan_data$gamma_shape, cv_stan_data[[theta_rate_param_name]])
      
      beta_est <- base::rep(NA, p_current)
      tryCatch({
        all_args_for_sample <- base::c(
          base::list(data = cv_stan_data, seed = seed + i*10 + k, init = init_fun_cv),
          cv_mcmc_params_worker
        )
        fit_cv <- base::do.call(stan_model_obj_for_cv$sample, all_args_for_sample)
        
        if (base::all(fit_cv$return_codes() == 0)) {
          beta_var_name <- if (kernel_name == "gaussian") "beta_params" else "beta"
          beta_draws <- fit_cv$draws(variables = beta_var_name, format = "df")
          beta_est <- base::colMeans(dplyr::select(beta_draws, dplyr::starts_with(paste0(beta_var_name,"["))))
        }
      }, error = function(e) {
        # CV fit might fail for bad h, which is expected
      })
      
      if (base::any(base::is.na(beta_est))) {
        fold_losses[k] <- Inf
      } else {
        fold_losses[k] <- base::mean(rho_tau_loss(y_cv_val - X_cv_val %*% beta_est, tau))
      }
    }
    avg_check_losses[i] <- base::mean(fold_losses, na.rm = TRUE)
  }
  
  best_h_idx <- base::which.min(avg_check_losses)
  if(base::length(best_h_idx) == 0 || base::is.infinite(avg_check_losses[best_h_idx]) || avg_check_losses[best_h_idx] == 0) {
    best_h_fallback_idx <- base::ceiling(base::length(h_grid)/2)
    best_h <- h_grid[best_h_fallback_idx]
    base::cat(base::paste0(log_prefix_cv, " CV could not find best h or loss was Inf/0, using median: ", best_h, " (from loss: ", avg_check_losses[best_h_fallback_idx],") @@@\n"))
  } else {
    best_h <- h_grid[best_h_idx]
    base::cat(base::paste0(log_prefix_cv, " CV best h found: ", best_h, " (loss: ", avg_check_losses[best_h_idx], ") @@@\n"))
  }
  base::cat(base::paste0(log_prefix_cv, " ending perform_cv_for_h for ", kernel_name, ". @@@\n"))
  return(best_h)
}


#### --- 4. Stan Model Files --- ####
stan_file_gaussian <- "bsqr_gaussian_Z_robust.stan"
stan_file_uniform <- "bsqr_uniform_Z_robust.stan"
stan_file_epanechnikov <- "bsqr_epanechnikov_Z_robust.stan"
stan_file_triangular <- "bsqr_triangular_Z_robust.stan"

stan_files_to_check <- c(stan_file_gaussian, stan_file_uniform, stan_file_epanechnikov, stan_file_triangular)
for (f in stan_files_to_check) {
  if (!file.exists(f)) {
    stop(paste("Stan model file '", f, "' not found in working directory: ", getwd()))
  }
}

#### --- 5. Compile All Stan Models (in Main Process) --- ####
compile_model <- function(file_name, model_name) {
  cat(paste("Compiling", model_name, "Stan model (", file_name, ") in main process (if needed)...\n")); flush.console()
  model_obj <- NULL
  tryCatch({
    model_obj <- cmdstanr::cmdstan_model(
      stan_file = file_name,
      quiet = FALSE,
      force_recompile = FALSE,
      compile_model_methods = TRUE
    )
    cat(model_name, "Stan model compilation/load successful.\n"); flush.console()
  }, error = function(e) {
    message(paste("Error during", model_name, "Stan model compilation:"))
    message(conditionMessage(e))
    stop(paste(model_name, "compilation failed. Fix before parallelizing."))
  })
  return(model_obj)
}

model_gaussian <- compile_model(stan_file_gaussian, "Gaussian")
model_uniform <- compile_model(stan_file_uniform, "Uniform")
model_epanechnikov <- compile_model(stan_file_epanechnikov, "Epanechnikov")
model_triangular <- compile_model(stan_file_triangular, "Triangular")
rm(model_gaussian, model_uniform, model_epanechnikov, model_triangular); gc()

#### --- Function to process a single replication (for parallelization) --- ####
run_single_replication <- function(m_rep_idx, p_config) {
  worker_pid <- Sys.getpid()
  log_prefix <- base::paste0("@@@ WORKER ", worker_pid, ", REP ", m_rep_idx, " (Err: ", p_config$current_error_spec$name_suffix,
                             ", Scen: ", p_config$scenario$name, ", Tau: ", p_config$tau_val_current, ")")
  
  base::cat(base::paste0(log_prefix, " --- START --- \n"))
  
  # --- Load/Compile all Stan models within the worker ---
  load_model_in_worker <- function(file_path, model_name) {
    base::cat(base::paste0(log_prefix, " - Compiling/Loading ", model_name, " model (", file_path, ") in worker...\n"))
    model_obj <- NULL
    tryCatch({
      model_obj <- cmdstanr::cmdstan_model(stan_file = file_path, quiet = FALSE, force_recompile = FALSE, compile_model_methods = TRUE)
      if (!is.null(model_obj)) base::cat(base::paste0(log_prefix, " - ", model_name, " model loaded in worker.\n"))
    }, error = function(e) {
      base::cat(base::paste0(log_prefix, " - ERROR loading ", model_name, " model in worker: ", base::conditionMessage(e), "\n"))
    })
    return(model_obj)
  }
  
  bsqr_model_g <- load_model_in_worker(p_config$stan_file_paths$gaussian, "Gaussian")
  bsqr_model_u <- load_model_in_worker(p_config$stan_file_paths$uniform, "Uniform")
  bsqr_model_e <- load_model_in_worker(p_config$stan_file_paths$epanechnikov, "Epanechnikov")
  bsqr_model_t <- load_model_in_worker(p_config$stan_file_paths$triangular, "Triangular")
  
  # --- Extract parameters from p_config ---
  p_current <- p_config$scenario$p
  beta_true_current <- p_config$scenario$beta_true
  Sigma_X_current <- p_config$Sigma_X_current
  tau_val_current <- p_config$tau_val_current
  
  # MCMC settings
  n_chains_mcmc_config <- p_config$n_chains_mcmc
  n_total_iter_mcmc_config <- p_config$n_total_iter_mcmc
  n_warmup_mcmc_config <- p_config$n_warmup_mcmc
  n_sampling_iter_mcmc_config <- p_config$n_sampling_iter_mcmc
  n_thin_mcmc_config <- p_config$n_thin_mcmc
  
  # CV settings
  USE_CV_FOR_H_local <- p_config$USE_CV_FOR_H
  CV_N_FOLDS_config <- p_config$CV_N_FOLDS
  CV_H_GRID_FACTORS_config <- p_config$CV_H_GRID_FACTORS
  CV_MCMC_PARAMS_orig_config <- p_config$CV_MCMC_PARAMS
  cv_mcmc_params_worker <- CV_MCMC_PARAMS_orig_config
  cv_mcmc_params_worker$parallel_chains <- base::min(CV_MCMC_PARAMS_orig_config$chains, p_config$CORES_PER_INNER_SAMPLER)
  
  # Prior and model settings
  beta_prior_mean_val_config <- p_config$beta_prior_mean_val
  beta_prior_var_val_config <- p_config$beta_prior_var_val
  theta_prior_a_r_config <- p_config$theta_prior_a_r
  theta_prior_b_r_config <- p_config$theta_prior_b_r
  upper_bound_theta_r_val_config <- p_config$upper_bound_theta_r_val
  epsilon_theta_val_config <- p_config$epsilon_theta_val
  Z_rel_tol_val_config <- p_config$Z_rel_tol_val
  K_ASYMPTOTIC_SWITCH_STD_DEVS_val_config <- p_config$K_ASYMPTOTIC_SWITCH_STD_DEVS_val
  USE_Z_ASYMPTOTIC_APPROX_val_config <- p_config$USE_Z_ASYMPTOTIC_APPROX_val
  Z_outer_integration_lower_bound_val_config <- p_config$Z_outer_integration_lower_bound_val
  Z_outer_integration_upper_bound_val_config <- p_config$Z_outer_integration_upper_bound_val
  sigma_ald_prior_scale_brms_config <- p_config$sigma_ald_prior_scale_brms
  beta_prior_sd_brms_config <- p_config$beta_prior_sd_brms
  
  # --- Initialize result storage for this replication ---
  # Initialize with NA_integer_ for counts that could be NA
  # RQ
  mse_beta_rq_rep <- NA_real_; mae_beta_rq_rep <- NA_real_; check_loss_test_rq_rep <- NA_real_; ME_rq_rep <- NA_real_
  bias_beta_rq_rep <- base::rep(NA_real_, p_current)
  # BQR-ALD
  mse_beta_bqr_ald_rep <- NA_real_; mae_beta_bqr_ald_rep <- NA_real_; check_loss_test_bqr_ald_rep <- NA_real_; ME_bqr_ald_rep <- NA_real_
  bias_beta_bqr_ald_rep <- base::rep(NA_real_, p_current); coverage_beta_bqr_ald_rep <- base::rep(NA_real_, p_current); ci_width_beta_bqr_ald_rep <- base::rep(NA_real_, p_current)
  fit_time_bqr_ald_rep <- NA_real_; rhat_beta_max_bqr_ald_rep <- NA_real_; ess_bulk_beta_min_bqr_ald_rep <- NA_real_; ess_tail_beta_min_bqr_ald_rep <- NA_real_
  rhat_sigma_ald_bqr_ald_rep <- NA_real_; ess_bulk_sigma_ald_bqr_ald_rep <- NA_real_; ess_tail_sigma_ald_bqr_ald_rep <- NA_real_
  num_divergences_bqr_ald_rep <- NA_integer_
  # BSQR-G
  mse_beta_bsqr_g_rep <- NA_real_; mae_beta_bsqr_g_rep <- NA_real_; check_loss_test_bsqr_g_rep <- NA_real_; ME_bsqr_g_rep <- NA_real_
  bias_beta_bsqr_g_rep <- base::rep(NA_real_, p_current); coverage_beta_bsqr_g_rep <- base::rep(NA_real_, p_current); ci_width_beta_bsqr_g_rep <- base::rep(NA_real_, p_current)
  fit_time_bsqr_g_rep <- NA_real_; rhat_beta_max_bsqr_g_rep <- NA_real_; ess_bulk_beta_min_bsqr_g_rep <- NA_real_; ess_tail_beta_min_bsqr_g_rep <- NA_real_
  rhat_theta_bsqr_g_rep <- NA_real_; ess_bulk_theta_bsqr_g_rep <- NA_real_; ess_tail_theta_bsqr_g_rep <- NA_real_
  num_divergences_bsqr_g_rep <- NA_integer_
  # BSQR-U
  mse_beta_bsqr_u_rep <- NA_real_; mae_beta_bsqr_u_rep <- NA_real_; check_loss_test_bsqr_u_rep <- NA_real_; ME_bsqr_u_rep <- NA_real_
  bias_beta_bsqr_u_rep <- base::rep(NA_real_, p_current); coverage_beta_bsqr_u_rep <- base::rep(NA_real_, p_current); ci_width_beta_bsqr_u_rep <- base::rep(NA_real_, p_current)
  fit_time_bsqr_u_rep <- NA_real_; rhat_beta_max_bsqr_u_rep <- NA_real_; ess_bulk_beta_min_bsqr_u_rep <- NA_real_; ess_tail_beta_min_bsqr_u_rep <- NA_real_
  rhat_theta_bsqr_u_rep <- NA_real_; ess_bulk_theta_bsqr_u_rep <- NA_real_; ess_tail_theta_bsqr_u_rep <- NA_real_
  num_divergences_bsqr_u_rep <- NA_integer_
  # BSQR-E
  mse_beta_bsqr_e_rep <- NA_real_; mae_beta_bsqr_e_rep <- NA_real_; check_loss_test_bsqr_e_rep <- NA_real_; ME_bsqr_e_rep <- NA_real_
  bias_beta_bsqr_e_rep <- base::rep(NA_real_, p_current); coverage_beta_bsqr_e_rep <- base::rep(NA_real_, p_current); ci_width_beta_bsqr_e_rep <- base::rep(NA_real_, p_current)
  fit_time_bsqr_e_rep <- NA_real_; rhat_beta_max_bsqr_e_rep <- NA_real_; ess_bulk_beta_min_bsqr_e_rep <- NA_real_; ess_tail_beta_min_bsqr_e_rep <- NA_real_
  rhat_theta_bsqr_e_rep <- NA_real_; ess_bulk_theta_bsqr_e_rep <- NA_real_; ess_tail_theta_bsqr_e_rep <- NA_real_
  num_divergences_bsqr_e_rep <- NA_integer_
  # BSQR-T
  mse_beta_bsqr_t_rep <- NA_real_; mae_beta_bsqr_t_rep <- NA_real_; check_loss_test_bsqr_t_rep <- NA_real_; ME_bsqr_t_rep <- NA_real_
  bias_beta_bsqr_t_rep <- base::rep(NA_real_, p_current); coverage_beta_bsqr_t_rep <- base::rep(NA_real_, p_current); ci_width_beta_bsqr_t_rep <- base::rep(NA_real_, p_current)
  fit_time_bsqr_t_rep <- NA_real_; rhat_beta_max_bsqr_t_rep <- NA_real_; ess_bulk_beta_min_bsqr_t_rep <- NA_real_; ess_tail_beta_min_bsqr_t_rep <- NA_real_
  rhat_theta_bsqr_t_rep <- NA_real_; ess_bulk_theta_bsqr_t_rep <- NA_real_; ess_tail_theta_bsqr_t_rep <- NA_real_
  num_divergences_bsqr_t_rep <- NA_integer_
  
  # --- Data Generation (Once per replication) ---
  current_loop_seed <- p_config$base_seed_mcmc + m_rep_idx * 10 + p_config$scenario_idx * 100 + p_config$tau_val_idx * 1000 + p_config$err_dist_idx * 10000
  base::cat(base::paste0(log_prefix, " - Setting seed: ", current_loop_seed, "\n"))
  base::set.seed(current_loop_seed)
  
  base::cat(base::paste0(log_prefix, " - Generating training and test data...\n"))
  data_train <- generate_data(p_config$n_train, p_current, beta_true_current, p_config$rho_x_val,
                              error_type = p_config$current_error_spec$type, error_params = p_config$current_error_spec$params)
  data_test  <- generate_data(p_config$n_test,  p_current, beta_true_current, p_config$rho_x_val,
                              error_type = p_config$current_error_spec$type, error_params = p_config$current_error_spec$params)
  
  X_train_orig <- data_train$X; y_train <- data_train$y
  X_test_orig  <- data_test$X; y_test  <- data_test$y
  base::cat(base::paste0(log_prefix, " - Data generation complete.\n"))
  
  # --- Model 1: Standard Quantile Regression (quantreg) ---
  base::cat(base::paste0(log_prefix, " - Fitting quantreg::rq...\n"))
  beta_hat_rq_current <- tryCatch({
    stats::coef(quantreg::rq(y_train ~ X_train_orig - 1, tau = tau_val_current, method = "br"))
  }, error = function(e) { base::rep(NA_real_, p_current) })
  
  if(!base::any(base::is.na(beta_hat_rq_current))){
    mse_beta_rq_rep <- base::mean((beta_hat_rq_current - beta_true_current)^2)
    mae_beta_rq_rep <- base::mean(base::abs(beta_hat_rq_current - beta_true_current))
    bias_beta_rq_rep <- beta_hat_rq_current - beta_true_current
    check_loss_test_rq_rep <- base::mean(rho_tau_loss(y_test - X_test_orig %*% beta_hat_rq_current, tau_val_current))
    ME_rq_rep <- base::as.numeric(base::t(bias_beta_rq_rep) %*% Sigma_X_current %*% bias_beta_rq_rep)
  }
  
  # --- Model 2: Bayesian QR with ALD (brms) ---
  base::cat(base::paste0(log_prefix, " - Fitting brms model (BQR-ALD)...\n"))
  X_train_brms <- data_train$X; base::colnames(X_train_brms) <- base::paste0("X", 1:p_current)
  X_test_brms <- data_test$X; base::colnames(X_test_brms) <- base::paste0("X", 1:p_current)
  brms_data_train <- base::data.frame(y_train = y_train, X_train_brms)
  bqr_ald_priors <- base::c(
    brms::set_prior(base::paste0("normal(0,", beta_prior_sd_brms_config, ")"), class = "b"),
    brms::set_prior(base::paste0("cauchy(0,", sigma_ald_prior_scale_brms_config, ")"), class = "sigma", lb = 0)
  )
  predictor_names_brms <- base::paste0("X", 1:p_current)
  brms_formula_str <- base::paste0("y_train ~ ", base::paste(predictor_names_brms, collapse = " + "), " -1")
  brms_formula_explicit <- brms::bf(stats::as.formula(brms_formula_str), quantile = tau_val_current)
  
  fit_bqr_ald_brms_model <- NULL; sampling_successful_bqr_ald <- FALSE
  bqr_ald_fit_start_time <- Sys.time()
  tryCatch({
    fit_bqr_ald_brms_model <- brms::brm(
      formula = brms_formula_explicit, data = brms_data_train,
      family = "asym_laplace", prior = bqr_ald_priors,
      chains = n_chains_mcmc_config, iter = n_total_iter_mcmc_config, warmup = n_warmup_mcmc_config, thin = n_thin_mcmc_config,
      seed = current_loop_seed + 2, cores = base::min(n_chains_mcmc_config, p_config$CORES_PER_INNER_SAMPLER),
      silent = 2, refresh = 0, backend = "cmdstanr", control = base::list(adapt_delta = 0.95)
    )
    if (!base::is.null(fit_bqr_ald_brms_model) && base::inherits(fit_bqr_ald_brms_model, "brmsfit") &&
        !base::is.null(fit_bqr_ald_brms_model$fit) && base::nrow(posterior::as_draws_df(fit_bqr_ald_brms_model)) > 0) {
      sampling_successful_bqr_ald <- TRUE
    }
  }, error = function(e) {
    base::cat(base::paste0(log_prefix, " - ERROR in brms fitting (BQR-ALD): ", base::conditionMessage(e), "\n"))
  })
  fit_time_bqr_ald_rep <- base::as.numeric(base::difftime(Sys.time(), bqr_ald_fit_start_time, units = "secs"))
  
  if(sampling_successful_bqr_ald) {
    draws_bqr_ald <- posterior::as_draws_df(fit_bqr_ald_brms_model)
    beta_draws_bqr_ald_df <- dplyr::select(draws_bqr_ald, dplyr::starts_with("b_X"))
    if (is.data.frame(beta_draws_bqr_ald_df) && ncol(beta_draws_bqr_ald_df) == p_current && nrow(beta_draws_bqr_ald_df) > 0) {
      beta_hat_bqr_ald <- base::colMeans(beta_draws_bqr_ald_df, na.rm = TRUE)
      mse_beta_bqr_ald_rep <- base::mean((beta_hat_bqr_ald - beta_true_current)^2, na.rm = TRUE)
      mae_beta_bqr_ald_rep <- base::mean(base::abs(beta_hat_bqr_ald - beta_true_current), na.rm = TRUE)
      bias_beta_bqr_ald_rep <- beta_hat_bqr_ald - beta_true_current
      ME_bqr_ald_rep <- base::as.numeric(base::t(bias_beta_bqr_ald_rep) %*% Sigma_X_current %*% bias_beta_bqr_ald_rep)
      tryCatch({
        posterior_preds_bqr_ald <- brms::posterior_epred(fit_bqr_ald_brms_model, newdata = base::as.data.frame(X_test_brms))
        y_pred_bqr_ald_point_estimate <- matrixStats::colQuantiles(posterior_preds_bqr_ald, probs = tau_val_current, na.rm = TRUE)
        check_loss_test_bqr_ald_rep <- base::mean(rho_tau_loss(y_test - y_pred_bqr_ald_point_estimate, tau_val_current), na.rm = TRUE)
      }, error = function(e_epred){ check_loss_test_bqr_ald_rep <- NA_real_ })
      beta_cis_bqr_ald_mat <- matrixStats::colQuantiles(as.matrix(beta_draws_bqr_ald_df), probs = c(0.025, 0.975), na.rm = TRUE)
      coverage_beta_bqr_ald_rep <- (beta_true_current >= beta_cis_bqr_ald_mat[, 1]) & (beta_true_current <= beta_cis_bqr_ald_mat[, 2])
      ci_width_beta_bqr_ald_rep <- beta_cis_bqr_ald_mat[, 2] - beta_cis_bqr_ald_mat[, 1]
    }
    summary_bqr_ald_posterior <- posterior::summarise_draws(fit_bqr_ald_brms_model)
    beta_vars_bqr_ald_names_posterior <- base::paste0("b_", predictor_names_brms)
    rhat_values_beta_bqr <- summary_bqr_ald_posterior$rhat[summary_bqr_ald_posterior$variable %in% beta_vars_bqr_ald_names_posterior]
    ess_bulk_values_beta_bqr <- summary_bqr_ald_posterior$ess_bulk[summary_bqr_ald_posterior$variable %in% beta_vars_bqr_ald_names_posterior]
    ess_tail_values_beta_bqr <- summary_bqr_ald_posterior$ess_tail[summary_bqr_ald_posterior$variable %in% beta_vars_bqr_ald_names_posterior]
    if(length(rhat_values_beta_bqr) > 0) rhat_beta_max_bqr_ald_rep <- base::max(rhat_values_beta_bqr, na.rm = TRUE)
    if(length(ess_bulk_values_beta_bqr) > 0) ess_bulk_beta_min_bqr_ald_rep <- base::min(ess_bulk_values_beta_bqr, na.rm = TRUE)
    if(length(ess_tail_values_beta_bqr) > 0) ess_tail_beta_min_bqr_ald_rep <- base::min(ess_tail_values_beta_bqr, na.rm = TRUE)
    sigma_ald_var_name <- "sigma"
    if (sigma_ald_var_name %in% summary_bqr_ald_posterior$variable) {
      rhat_sigma_ald_bqr_ald_rep <- summary_bqr_ald_posterior$rhat[summary_bqr_ald_posterior$variable == sigma_ald_var_name][1]
      ess_bulk_sigma_ald_bqr_ald_rep <- summary_bqr_ald_posterior$ess_bulk[summary_bqr_ald_posterior$variable == sigma_ald_var_name][1]
      ess_tail_sigma_ald_bqr_ald_rep <- summary_bqr_ald_posterior$ess_tail[summary_bqr_ald_posterior$variable == sigma_ald_var_name][1]
    }
    
    # Robust Divergence Extraction Logic (from your code)
    num_divergences_bqr_ald_rep <- NA_integer_
    if (inherits(fit_bqr_ald_brms_model, "brmsfit") && !is.null(fit_bqr_ald_brms_model$fit)) {
      if (inherits(fit_bqr_ald_brms_model$fit, "CmdStanMCMC")) {
        diag_summary_brms <- tryCatch({ diag_summary_brms <- fit_bqr_ald_brms_model$fit$diagnostic_summary(quiet = TRUE) }, error = function(e_ds) { NULL })
        if (!is.null(diag_summary_brms)) {
          if ("num_divergent" %in% names(diag_summary_brms) && !is.null(diag_summary_brms$num_divergent) && !all(is.na(diag_summary_brms$num_divergent))) {
            num_divergences_bqr_ald_rep <- sum(as.integer(diag_summary_brms$num_divergent), na.rm = TRUE)
          } else if ("sum_divergent" %in% names(diag_summary_brms) && !is.null(diag_summary_brms$sum_divergent) && !is.na(diag_summary_brms$sum_divergent)) {
            num_divergences_bqr_ald_rep <- as.integer(diag_summary_brms$sum_divergent)
          }
        }
      } else if (inherits(fit_bqr_ald_brms_model$fit, "stanfit")) {
        tryCatch({
          sampler_params_rstan_list <- rstan::get_sampler_params(fit_bqr_ald_brms_model$fit, inc_warmup = FALSE)
          if (is.list(sampler_params_rstan_list) && length(sampler_params_rstan_list) > 0) {
            divergences_per_chain <- sapply(sampler_params_rstan_list, function(chain_params) {
              if (is.matrix(chain_params) && "divergent__" %in% colnames(chain_params)) sum(chain_params[, "divergent__"], na.rm = TRUE) else NA_integer_
            })
            if (!all(is.na(divergences_per_chain))) num_divergences_bqr_ald_rep <- sum(divergences_per_chain, na.rm = TRUE)
          }
        }, error = function(e_gsp) {})
      } else {
        nuts_params_brms <- tryCatch({ nuts_params_brms <- brms::nuts_params(fit_bqr_ald_brms_model) }, error = function(e_np) { NULL })
        if (!is.null(nuts_params_brms) && "divergent__" %in% colnames(nuts_params_brms)) {
          num_divergences_bqr_ald_rep <- sum(nuts_params_brms$divergent__, na.rm = TRUE)
        }
      }
    }
  }
  
  # --- Common setup for BSQR models ---
  initial_residuals <- if(!base::any(base::is.na(beta_hat_rq_current))) y_train - X_train_orig %*% beta_hat_rq_current else y_train
  h_silverman_base <- calculate_h_silverman(initial_residuals, p_config$n_train)
  
  run_bsqr_kernel <- function(kernel_name, stan_model_obj) {
    base::cat(base::paste0(log_prefix, " - Starting BSQR with ", kernel_name, " kernel...\n"))
    
    # Initialize results for this kernel
    p_current_local <- p_config$scenario$p
    results_out <- list(mse=NA, mae=NA, loss=NA, me=NA, bias=rep(NA,p_current_local), cov=rep(NA,p_current_local), width=rep(NA,p_current_local), time=NA,
                        rhat_b=NA, ess_b=NA, ess_t=NA, rhat_th=NA, ess_b_th=NA, ess_t_th=NA, divs=NA_integer_)
    
    if (base::is.null(stan_model_obj)) {
      base::cat(base::paste0(log_prefix, " - Skipping BSQR-", kernel_name, " because Stan model is invalid.\n"))
      return(results_out)
    }
    
    # 1. Select h via CV or Silverman
    h_val_current <- if (USE_CV_FOR_H_local == 1) {
      h_grid_current <- h_silverman_base * CV_H_GRID_FACTORS_config
      base_stan_data_cv <- list(
        K = p_current_local, tau = tau_val_current,
        beta_location = base::rep(beta_prior_mean_val_config, p_current_local), beta_scale = base::rep(base::sqrt(beta_prior_var_val_config), p_current_local),
        gamma_shape = theta_prior_a_r_config, upper_bound_for_theta = upper_bound_theta_r_val_config, epsilon_theta = base::max(epsilon_theta_val_config, 1e-7)
      )
      if (kernel_name == "gaussian") { base_stan_data_cv <- c(base_stan_data_cv, list(theta_prior_rate_val = theta_prior_b_r_config, Z_rel_tol = Z_rel_tol_val_config, K_ASYMPTOTIC_SWITCH_STD_DEVS = K_ASYMPTOTIC_SWITCH_STD_DEVS_val_config, USE_Z_ASYMPTOTIC_APPROX = USE_Z_ASYMPTOTIC_APPROX_val_config))
      } else if (kernel_name == "uniform") { base_stan_data_cv$theta_prior_rate_val = theta_prior_b_r_config
      } else { # Epan and Triangular
        base_stan_data_cv <- c(base_stan_data_cv, list(gamma_rate = theta_prior_b_r_config, Z_rel_tol = Z_rel_tol_val_config, K_ASYMPTOTIC_SWITCH_STD_DEVS = K_ASYMPTOTIC_SWITCH_STD_DEVS_val_config, USE_Z_ASYMPTOTIC_APPROX = USE_Z_ASYMPTOTIC_APPROX_val_config))
        if (kernel_name == "epanechnikov") { 
          base_stan_data_cv <- c(base_stan_data_cv, list(Z_outer_integration_lower_bound = Z_outer_integration_lower_bound_val_config, Z_outer_integration_upper_bound = Z_outer_integration_upper_bound_val_config)) 
        }
      }
      perform_cv_for_h(X_train_orig, y_train, tau_val_current, h_grid_current, CV_N_FOLDS_config, stan_model_obj, kernel_name, cv_mcmc_params_worker, base_stan_data_cv, current_loop_seed, log_prefix)
    } else {
      h_silverman_base
    }
    if(is.na(h_val_current) || h_val_current <= 0) h_val_current <- 0.1
    
    # 2. Prepare main Stan data
    main_stan_data <- list(
      N_train_obs = p_config$n_train, K = p_current_local, X_train = X_train_orig, y_train = y_train, tau = tau_val_current, h = h_val_current,
      beta_location = base::rep(beta_prior_mean_val_config, p_current_local), beta_scale = base::rep(base::sqrt(beta_prior_var_val_config), p_current_local),
      gamma_shape = theta_prior_a_r_config, upper_bound_for_theta = upper_bound_theta_r_val_config, epsilon_theta = base::max(epsilon_theta_val_config, 1e-7)
    )
    if (kernel_name == "gaussian") { main_stan_data <- c(main_stan_data, list(theta_prior_rate_val = theta_prior_b_r_config, Z_rel_tol = Z_rel_tol_val_config, K_ASYMPTOTIC_SWITCH_STD_DEVS = K_ASYMPTOTIC_SWITCH_STD_DEVS_val_config, USE_Z_ASYMPTOTIC_APPROX = USE_Z_ASYMPTOTIC_APPROX_val_config))
    } else if (kernel_name == "uniform") { main_stan_data$theta_prior_rate_val = theta_prior_b_r_config
    } else { # Epan and Triangular
      main_stan_data <- c(main_stan_data, list(gamma_rate = theta_prior_b_r_config, Z_rel_tol = Z_rel_tol_val_config, K_ASYMPTOTIC_SWITCH_STD_DEVS = K_ASYMPTOTIC_SWITCH_STD_DEVS_val_config, USE_Z_ASYMPTOTIC_APPROX = USE_Z_ASYMPTOTIC_APPROX_val_config))
      if (kernel_name == "epanechnikov") { 
        main_stan_data <- c(main_stan_data, list(Z_outer_integration_lower_bound = Z_outer_integration_lower_bound_val_config, Z_outer_integration_upper_bound = Z_outer_integration_upper_bound_val_config)) 
      }
    }
    theta_rate_val_for_init <- if (kernel_name %in% c("epanechnikov", "triangular")) main_stan_data$gamma_rate else main_stan_data$theta_prior_rate_val
    init_fun <- function() provide_inits(p_current_local, main_stan_data$epsilon_theta, main_stan_data$upper_bound_for_theta, main_stan_data$gamma_shape, theta_rate_val_for_init)
    
    # 3. Fit model
    fit_obj <- NULL
    fit_start_time <- Sys.time()
    tryCatch({
      fit_obj <- stan_model_obj$sample(
        data = main_stan_data, seed = current_loop_seed,
        chains = n_chains_mcmc_config, parallel_chains = base::min(n_chains_mcmc_config, p_config$CORES_PER_INNER_SAMPLER),
        iter_warmup = n_warmup_mcmc_config, iter_sampling = n_sampling_iter_mcmc_config, thin = n_thin_mcmc_config,
        refresh = 0, show_messages = FALSE, init = init_fun, adapt_delta = 0.99, max_treedepth = 12
      )
    }, error=function(e){ base::cat(base::paste0(log_prefix, " - ERROR fitting BSQR-", kernel_name, ": ", conditionMessage(e), "\n"))})
    results_out$time <- as.numeric(difftime(Sys.time(), fit_start_time, units = "secs"))
    
    # 4. Process results
    if (!is.null(fit_obj) && all(fit_obj$return_codes() == 0)) {
      beta_var_name <- if (kernel_name == "gaussian") "beta_params" else "beta"
      draws_df <- fit_obj$draws(variables = c(beta_var_name, "theta"), format = "df")
      beta_draws_df <- dplyr::select(draws_df, dplyr::starts_with(paste0(beta_var_name, "[")))
      
      beta_hat <- base::colMeans(beta_draws_df, na.rm=TRUE)
      results_out$mse <- base::mean((beta_hat - beta_true_current)^2)
      results_out$mae <- base::mean(base::abs(beta_hat - beta_true_current))
      results_out$bias <- beta_hat - beta_true_current
      results_out$loss <- base::mean(rho_tau_loss(y_test - X_test_orig %*% beta_hat, tau_val_current))
      results_out$me <- base::as.numeric(base::t(results_out$bias) %*% Sigma_X_current %*% results_out$bias)
      
      cis_mat <- matrixStats::colQuantiles(as.matrix(beta_draws_df), probs=c(0.025, 0.975), na.rm=TRUE)
      results_out$cov <- (beta_true_current >= cis_mat[,1]) & (beta_true_current <= cis_mat[,2])
      results_out$width <- cis_mat[,2] - cis_mat[,1]
      
      summary_stan <- fit_obj$summary(variables = c(beta_var_name, "theta"))
      beta_indices_in_summary <- grep(paste0("^", beta_var_name), summary_stan$variable)
      results_out$rhat_b <- max(summary_stan$rhat[beta_indices_in_summary], na.rm=TRUE)
      results_out$ess_b <- min(summary_stan$ess_bulk[beta_indices_in_summary], na.rm=TRUE)
      results_out$ess_t <- min(summary_stan$ess_tail[beta_indices_in_summary], na.rm=TRUE)
      
      theta_index_in_summary <- which(summary_stan$variable == "theta")
      results_out$rhat_th <- summary_stan$rhat[theta_index_in_summary]
      results_out$ess_b_th <- summary_stan$ess_bulk[theta_index_in_summary]
      results_out$ess_t_th <- summary_stan$ess_tail[theta_index_in_summary]
      
      # Robust Divergence Extraction (from your code)
      diag_summary_bsqr <- tryCatch({ diag_summary_bsqr <- fit_obj$diagnostic_summary(quiet = TRUE) }, error = function(e_ds_g) { NULL })
      if (!is.null(diag_summary_bsqr)) {
        if ("num_divergent" %in% names(diag_summary_bsqr) && !is.null(diag_summary_bsqr$num_divergent) && !all(is.na(diag_summary_bsqr$num_divergent))) {
          results_out$divs <- sum(as.integer(diag_summary_bsqr$num_divergent), na.rm = TRUE)
        } else if ("sum_divergent" %in% names(diag_summary_bsqr) && !is.null(diag_summary_bsqr$sum_divergent) && !is.na(diag_summary_bsqr$sum_divergent)) {
          results_out$divs <- as.integer(diag_summary_bsqr$sum_divergent)
        }
      }
    }
    return(results_out)
  }
  
  # --- Run all BSQR models ---
  res_g <- run_bsqr_kernel("gaussian", bsqr_model_g)
  mse_beta_bsqr_g_rep <- res_g$mse; mae_beta_bsqr_g_rep <- res_g$mae; check_loss_test_bsqr_g_rep <- res_g$loss; ME_bsqr_g_rep <- res_g$me; bias_beta_bsqr_g_rep <- res_g$bias; coverage_beta_bsqr_g_rep <- res_g$cov; ci_width_beta_bsqr_g_rep <- res_g$width; fit_time_bsqr_g_rep <- res_g$time; rhat_beta_max_bsqr_g_rep <- res_g$rhat_b; ess_bulk_beta_min_bsqr_g_rep <- res_g$ess_b; ess_tail_beta_min_bsqr_g_rep <- res_g$ess_t; rhat_theta_bsqr_g_rep <- res_g$rhat_th; ess_bulk_theta_bsqr_g_rep <- res_g$ess_b_th; ess_tail_theta_bsqr_g_rep <- res_g$ess_t_th; num_divergences_bsqr_g_rep <- res_g$divs
  
  res_u <- run_bsqr_kernel("uniform", bsqr_model_u)
  mse_beta_bsqr_u_rep <- res_u$mse; mae_beta_bsqr_u_rep <- res_u$mae; check_loss_test_bsqr_u_rep <- res_u$loss; ME_bsqr_u_rep <- res_u$me; bias_beta_bsqr_u_rep <- res_u$bias; coverage_beta_bsqr_u_rep <- res_u$cov; ci_width_beta_bsqr_u_rep <- res_u$width; fit_time_bsqr_u_rep <- res_u$time; rhat_beta_max_bsqr_u_rep <- res_u$rhat_b; ess_bulk_beta_min_bsqr_u_rep <- res_u$ess_b; ess_tail_beta_min_bsqr_u_rep <- res_u$ess_t; rhat_theta_bsqr_u_rep <- res_u$rhat_th; ess_bulk_theta_bsqr_u_rep <- res_u$ess_b_th; ess_tail_theta_bsqr_u_rep <- res_u$ess_t_th; num_divergences_bsqr_u_rep <- res_u$divs
  
  res_e <- run_bsqr_kernel("epanechnikov", bsqr_model_e)
  mse_beta_bsqr_e_rep <- res_e$mse; mae_beta_bsqr_e_rep <- res_e$mae; check_loss_test_bsqr_e_rep <- res_e$loss; ME_bsqr_e_rep <- res_e$me; bias_beta_bsqr_e_rep <- res_e$bias; coverage_beta_bsqr_e_rep <- res_e$cov; ci_width_beta_bsqr_e_rep <- res_e$width; fit_time_bsqr_e_rep <- res_e$time; rhat_beta_max_bsqr_e_rep <- res_e$rhat_b; ess_bulk_beta_min_bsqr_e_rep <- res_e$ess_b; ess_tail_beta_min_bsqr_e_rep <- res_e$ess_t; rhat_theta_bsqr_e_rep <- res_e$rhat_th; ess_bulk_theta_bsqr_e_rep <- res_e$ess_b_th; ess_tail_theta_bsqr_e_rep <- res_e$ess_t_th; num_divergences_bsqr_e_rep <- res_e$divs
  
  res_t <- run_bsqr_kernel("triangular", bsqr_model_t)
  mse_beta_bsqr_t_rep <- res_t$mse; mae_beta_bsqr_t_rep <- res_t$mae; check_loss_test_bsqr_t_rep <- res_t$loss; ME_bsqr_t_rep <- res_t$me; bias_beta_bsqr_t_rep <- res_t$bias; coverage_beta_bsqr_t_rep <- res_t$cov; ci_width_beta_bsqr_t_rep <- res_t$width; fit_time_bsqr_t_rep <- res_t$time; rhat_beta_max_bsqr_t_rep <- res_t$rhat_b; ess_bulk_beta_min_bsqr_t_rep <- res_t$ess_b; ess_tail_beta_min_bsqr_t_rep <- res_t$ess_t; rhat_theta_bsqr_t_rep <- res_t$rhat_th; ess_bulk_theta_bsqr_t_rep <- res_t$ess_b_th; ess_tail_theta_bsqr_t_rep <- res_t$ess_t_th; num_divergences_bsqr_t_rep <- res_t$divs
  
  base::gc()
  base::cat(base::paste0(log_prefix, " --- END --- \n"))
  
  return(base::list(
    # RQ
    mse_beta_rq = mse_beta_rq_rep, mae_beta_rq = mae_beta_rq_rep, check_loss_test_rq = check_loss_test_rq_rep, ME_rq = ME_rq_rep, bias_beta_rq = bias_beta_rq_rep,
    # BQR-ALD
    mse_beta_bqr_ald = mse_beta_bqr_ald_rep, mae_beta_bqr_ald = mae_beta_bqr_ald_rep, check_loss_test_bqr_ald = check_loss_test_bqr_ald_rep, ME_bqr_ald = ME_bqr_ald_rep, bias_beta_bqr_ald = bias_beta_bqr_ald_rep, coverage_beta_bqr_ald = coverage_beta_bqr_ald_rep, ci_width_beta_bqr_ald = ci_width_beta_bqr_ald_rep, fit_time_bqr_ald = fit_time_bqr_ald_rep, rhat_beta_max_bqr_ald = rhat_beta_max_bqr_ald_rep, ess_bulk_beta_min_bqr_ald = ess_bulk_beta_min_bqr_ald_rep, ess_tail_beta_min_bqr_ald = ess_tail_beta_min_bqr_ald_rep, rhat_sigma_ald_bqr_ald = rhat_sigma_ald_bqr_ald_rep, ess_bulk_sigma_ald_bqr_ald = ess_bulk_sigma_ald_bqr_ald_rep, ess_tail_sigma_ald_bqr_ald = ess_tail_sigma_ald_bqr_ald_rep, num_divergences_bqr_ald = num_divergences_bqr_ald_rep,
    # BSQR-G
    mse_beta_bsqr_g = mse_beta_bsqr_g_rep, mae_beta_bsqr_g = mae_beta_bsqr_g_rep, check_loss_test_bsqr_g = check_loss_test_bsqr_g_rep, ME_bsqr_g = ME_bsqr_g_rep, bias_beta_bsqr_g = bias_beta_bsqr_g_rep, coverage_beta_bsqr_g = coverage_beta_bsqr_g_rep, ci_width_beta_bsqr_g = ci_width_beta_bsqr_g_rep, fit_time_bsqr_g = fit_time_bsqr_g_rep, rhat_beta_max_bsqr_g = rhat_beta_max_bsqr_g_rep, ess_bulk_beta_min_bsqr_g = ess_bulk_beta_min_bsqr_g_rep, ess_tail_beta_min_bsqr_g = ess_tail_beta_min_bsqr_g_rep, rhat_theta_bsqr_g = rhat_theta_bsqr_g_rep, ess_bulk_theta_bsqr_g = ess_bulk_theta_bsqr_g_rep, ess_tail_theta_bsqr_g = ess_tail_theta_bsqr_g_rep, num_divergences_bsqr_g = num_divergences_bsqr_g_rep,
    # BSQR-U
    mse_beta_bsqr_u = mse_beta_bsqr_u_rep, mae_beta_bsqr_u = mae_beta_bsqr_u_rep, check_loss_test_bsqr_u = check_loss_test_bsqr_u_rep, ME_bsqr_u = ME_bsqr_u_rep, bias_beta_bsqr_u = bias_beta_bsqr_u_rep, coverage_beta_bsqr_u = coverage_beta_bsqr_u_rep, ci_width_beta_bsqr_u = ci_width_beta_bsqr_u_rep, fit_time_bsqr_u = fit_time_bsqr_u_rep, rhat_beta_max_bsqr_u = rhat_beta_max_bsqr_u_rep, ess_bulk_beta_min_bsqr_u = ess_bulk_beta_min_bsqr_u_rep, ess_tail_beta_min_bsqr_u = ess_tail_beta_min_bsqr_u_rep, rhat_theta_bsqr_u = rhat_theta_bsqr_u_rep, ess_bulk_theta_bsqr_u = ess_bulk_theta_bsqr_u_rep, ess_tail_theta_bsqr_u = ess_tail_theta_bsqr_u_rep, num_divergences_bsqr_u = num_divergences_bsqr_u_rep,
    # BSQR-E
    mse_beta_bsqr_e = mse_beta_bsqr_e_rep, mae_beta_bsqr_e = mae_beta_bsqr_e_rep, check_loss_test_bsqr_e = check_loss_test_bsqr_e_rep, ME_bsqr_e = ME_bsqr_e_rep, bias_beta_bsqr_e = bias_beta_bsqr_e_rep, coverage_beta_bsqr_e = coverage_beta_bsqr_e_rep, ci_width_beta_bsqr_e = ci_width_beta_bsqr_e_rep, fit_time_bsqr_e = fit_time_bsqr_e_rep, rhat_beta_max_bsqr_e = rhat_beta_max_bsqr_e_rep, ess_bulk_beta_min_bsqr_e = ess_bulk_beta_min_bsqr_e_rep, ess_tail_beta_min_bsqr_e = ess_tail_beta_min_bsqr_e_rep, rhat_theta_bsqr_e = rhat_theta_bsqr_e_rep, ess_bulk_theta_bsqr_e = ess_bulk_theta_bsqr_e_rep, ess_tail_theta_bsqr_e = ess_tail_theta_bsqr_e_rep, num_divergences_bsqr_e = num_divergences_bsqr_e_rep,
    # BSQR-T
    mse_beta_bsqr_t = mse_beta_bsqr_t_rep, mae_beta_bsqr_t = mae_beta_bsqr_t_rep, check_loss_test_bsqr_t = check_loss_test_bsqr_t_rep, ME_bsqr_t = ME_bsqr_t_rep, bias_beta_bsqr_t = bias_beta_bsqr_t_rep, coverage_beta_bsqr_t = coverage_beta_bsqr_t_rep, ci_width_beta_bsqr_t = ci_width_beta_bsqr_t_rep, fit_time_bsqr_t = fit_time_bsqr_t_rep, rhat_beta_max_bsqr_t = rhat_beta_max_bsqr_t_rep, ess_bulk_beta_min_bsqr_t = ess_bulk_beta_min_bsqr_t_rep, ess_tail_beta_min_bsqr_t = ess_tail_beta_min_bsqr_t_rep, rhat_theta_bsqr_t = rhat_theta_bsqr_t_rep, ess_bulk_theta_bsqr_t = ess_bulk_theta_bsqr_t_rep, ess_tail_theta_bsqr_t = ess_tail_theta_bsqr_t_rep, num_divergences_bsqr_t = num_divergences_bsqr_t_rep
  ))
}

#### --- 6. Main Simulation Loop --- ####
future::plan(future::multisession, workers = N_WORKERS_OUTER)
options(future.apply.debug = TRUE)
on.exit(options(future.apply.debug = FALSE), add = TRUE)

all_results_list <- list()
overall_start_time <- Sys.time()
base_seed_mcmc <- 45678

for (err_dist_idx in 1:length(error_distributions)) {
  current_error_spec <- error_distributions[[err_dist_idx]]
  for (scenario_idx in 1:length(beta_scenarios)) {
    scenario <- beta_scenarios[[scenario_idx]]
    p_current <- scenario$p
    Sigma_X_current <- base::matrix(0, nrow = p_current, ncol = p_current)
    for(j in 1:p_current) for(k in 1:p_current) Sigma_X_current[j,k] <- rho_x_val^abs(j-k)
    
    for (tau_val_idx in 1:length(taus_sim)) {
      tau_val_current <- taus_sim[tau_val_idx]
      
      cat(paste0("\n\n=== STARTING CONFIG: Err=", current_error_spec$name_suffix, 
                 ", Scen=", scenario$name, ", Tau=", tau_val_current, " ===\n")); flush.console()
      
      # Initialize storage for this configuration
      # RQ
      mse_beta_rq_all_reps <- mae_beta_rq_all_reps <- check_loss_test_rq_all_reps <- ME_rq_all_reps <- base::numeric(M)
      bias_beta_rq_list_all_reps <- base::vector("list", M)
      # BQR-ALD
      mse_beta_bqr_ald_all_reps <- mae_beta_bqr_ald_all_reps <- check_loss_test_bqr_ald_all_reps <- ME_bqr_ald_all_reps <- base::numeric(M)
      bias_beta_bqr_ald_list_all_reps <- base::vector("list", M); coverage_beta_bqr_ald_list_all_reps <- ci_width_beta_bqr_ald_list_all_reps <- base::vector("list", M)
      fit_time_bqr_ald_all_reps <- rhat_beta_max_bqr_ald_all_reps <- ess_bulk_beta_min_bqr_ald_all_reps <- ess_tail_beta_min_bqr_ald_all_reps <- base::numeric(M)
      rhat_sigma_ald_bqr_ald_all_reps <- ess_bulk_sigma_ald_bqr_ald_all_reps <- ess_tail_sigma_ald_bqr_ald_all_reps <- base::numeric(M)
      num_divergences_bqr_ald_all_reps <- base::numeric(M)
      # BSQR-G
      mse_beta_bsqr_g_all_reps <- mae_beta_bsqr_g_all_reps <- check_loss_test_bsqr_g_all_reps <- ME_bsqr_g_all_reps <- base::numeric(M)
      bias_beta_bsqr_g_list_all_reps <- base::vector("list", M); coverage_beta_bsqr_g_list_all_reps <- ci_width_beta_bsqr_g_list_all_reps <- base::vector("list", M)
      fit_time_bsqr_g_all_reps <- rhat_beta_max_bsqr_g_all_reps <- ess_bulk_beta_min_bsqr_g_all_reps <- ess_tail_beta_min_bsqr_g_all_reps <- base::numeric(M)
      rhat_theta_bsqr_g_all_reps <- ess_bulk_theta_bsqr_g_all_reps <- ess_tail_theta_bsqr_g_all_reps <- base::numeric(M)
      num_divergences_bsqr_g_all_reps <- base::numeric(M)
      # BSQR-U
      mse_beta_bsqr_u_all_reps <- mae_beta_bsqr_u_all_reps <- check_loss_test_bsqr_u_all_reps <- ME_bsqr_u_all_reps <- base::numeric(M)
      bias_beta_bsqr_u_list_all_reps <- base::vector("list", M); coverage_beta_bsqr_u_list_all_reps <- ci_width_beta_bsqr_u_list_all_reps <- base::vector("list", M)
      fit_time_bsqr_u_all_reps <- rhat_beta_max_bsqr_u_all_reps <- ess_bulk_beta_min_bsqr_u_all_reps <- ess_tail_beta_min_bsqr_u_all_reps <- base::numeric(M)
      rhat_theta_bsqr_u_all_reps <- ess_bulk_theta_bsqr_u_all_reps <- ess_tail_theta_bsqr_u_all_reps <- base::numeric(M)
      num_divergences_bsqr_u_all_reps <- base::numeric(M)
      # BSQR-E
      mse_beta_bsqr_e_all_reps <- mae_beta_bsqr_e_all_reps <- check_loss_test_bsqr_e_all_reps <- ME_bsqr_e_all_reps <- base::numeric(M)
      bias_beta_bsqr_e_list_all_reps <- base::vector("list", M); coverage_beta_bsqr_e_list_all_reps <- ci_width_beta_bsqr_e_list_all_reps <- base::vector("list", M)
      fit_time_bsqr_e_all_reps <- rhat_beta_max_bsqr_e_all_reps <- ess_bulk_beta_min_bsqr_e_all_reps <- ess_tail_beta_min_bsqr_e_all_reps <- base::numeric(M)
      rhat_theta_bsqr_e_all_reps <- ess_bulk_theta_bsqr_e_all_reps <- ess_tail_theta_bsqr_e_all_reps <- base::numeric(M)
      num_divergences_bsqr_e_all_reps <- base::numeric(M)
      # BSQR-T
      mse_beta_bsqr_t_all_reps <- mae_beta_bsqr_t_all_reps <- check_loss_test_bsqr_t_all_reps <- ME_bsqr_t_all_reps <- base::numeric(M)
      bias_beta_bsqr_t_list_all_reps <- base::vector("list", M); coverage_beta_bsqr_t_list_all_reps <- ci_width_beta_bsqr_t_list_all_reps <- base::vector("list", M)
      fit_time_bsqr_t_all_reps <- rhat_beta_max_bsqr_t_all_reps <- ess_bulk_beta_min_bsqr_t_all_reps <- ess_tail_beta_min_bsqr_t_all_reps <- base::numeric(M)
      rhat_theta_bsqr_t_all_reps <- ess_bulk_theta_bsqr_t_all_reps <- ess_tail_theta_bsqr_t_all_reps <- base::numeric(M)
      num_divergences_bsqr_t_all_reps <- base::numeric(M)
      
      rep_start_time_tau <- Sys.time()
      
      config_for_replications <- list(
        current_error_spec = current_error_spec, scenario = scenario, tau_val_current = tau_val_current, Sigma_X_current = Sigma_X_current,
        err_dist_idx = err_dist_idx, scenario_idx = scenario_idx, tau_val_idx = tau_val_idx,
        n_train = n_train, n_test = n_test, rho_x_val = rho_x_val, base_seed_mcmc = base_seed_mcmc,
        n_chains_mcmc = n_chains_mcmc, n_total_iter_mcmc = n_total_iter_mcmc, n_warmup_mcmc = n_warmup_mcmc, n_sampling_iter_mcmc = n_sampling_iter_mcmc, n_thin_mcmc = n_thin_mcmc,
        CORES_PER_INNER_SAMPLER = CORES_PER_INNER_SAMPLER,
        USE_CV_FOR_H = USE_CV_FOR_H, CV_N_FOLDS = CV_N_FOLDS, CV_H_GRID_FACTORS = CV_H_GRID_FACTORS, CV_MCMC_PARAMS = CV_MCMC_PARAMS,
        beta_prior_mean_val = beta_prior_mean_val, beta_prior_var_val = beta_prior_var_val,
        theta_prior_a_r = theta_prior_a_r, theta_prior_b_r = theta_prior_b_r,
        upper_bound_theta_r_val = upper_bound_theta_r_val, epsilon_theta_val = epsilon_theta_val, Z_rel_tol_val = Z_rel_tol_val,
        K_ASYMPTOTIC_SWITCH_STD_DEVS_val = K_ASYMPTOTIC_SWITCH_STD_DEVS_val, USE_Z_ASYMPTOTIC_APPROX_val = USE_Z_ASYMPTOTIC_APPROX_val,
        Z_outer_integration_lower_bound_val = Z_outer_integration_lower_bound_val,
        Z_outer_integration_upper_bound_val = Z_outer_integration_upper_bound_val,
        sigma_ald_prior_scale_brms = sigma_ald_prior_scale_brms, beta_prior_sd_brms = beta_prior_sd_brms,
        stan_file_paths = list(
          gaussian = normalizePath(stan_file_gaussian), uniform = normalizePath(stan_file_uniform),
          epanechnikov = normalizePath(stan_file_epanechnikov), triangular = normalizePath(stan_file_triangular)
        )
      )
      
      progressr::with_progress({
        p_bar <- progressr::progressor(steps = M)
        all_reps_results <- future.apply::future_lapply(
          X = 1:M,
          FUN = function(m, p_config_fun) {
            res_single <- run_single_replication(m_rep_idx = m, p_config = p_config_fun)
            p_bar()
            return(res_single)
          },
          p_config_fun = config_for_replications,
          future.seed = TRUE, future.packages = required_packages,
          future.stdout = TRUE, future.conditions = "always"
        )
      })
      
      # Process results from all replications
      for (m_idx in 1:M) {
        rep_res <- all_reps_results[[m_idx]]
        if (is.list(rep_res)) {
          # RQ
          mse_beta_rq_all_reps[m_idx] <- rep_res$mse_beta_rq; mae_beta_rq_all_reps[m_idx] <- rep_res$mae_beta_rq; check_loss_test_rq_all_reps[m_idx] <- rep_res$check_loss_test_rq; ME_rq_all_reps[m_idx] <- rep_res$ME_rq; bias_beta_rq_list_all_reps[[m_idx]] <- rep_res$bias_beta_rq
          # BQR_ALD
          mse_beta_bqr_ald_all_reps[m_idx] <- rep_res$mse_beta_bqr_ald; mae_beta_bqr_ald_all_reps[m_idx] <- rep_res$mae_beta_bqr_ald; check_loss_test_bqr_ald_all_reps[m_idx] <- rep_res$check_loss_test_bqr_ald; ME_bqr_ald_all_reps[m_idx] <- rep_res$ME_bqr_ald; bias_beta_bqr_ald_list_all_reps[[m_idx]] <- rep_res$bias_beta_bqr_ald; coverage_beta_bqr_ald_list_all_reps[[m_idx]] <- rep_res$coverage_beta_bqr_ald; ci_width_beta_bqr_ald_list_all_reps[[m_idx]] <- rep_res$ci_width_beta_bqr_ald; fit_time_bqr_ald_all_reps[m_idx] <- rep_res$fit_time_bqr_ald; rhat_beta_max_bqr_ald_all_reps[m_idx] <- rep_res$rhat_beta_max_bqr_ald; ess_bulk_beta_min_bqr_ald_all_reps[m_idx] <- rep_res$ess_bulk_beta_min_bqr_ald; ess_tail_beta_min_bqr_ald_all_reps[m_idx] <- rep_res$ess_tail_beta_min_bqr_ald; rhat_sigma_ald_bqr_ald_all_reps[m_idx] <- rep_res$rhat_sigma_ald_bqr_ald; ess_bulk_sigma_ald_bqr_ald_all_reps[m_idx] <- rep_res$ess_bulk_sigma_ald_bqr_ald; ess_tail_sigma_ald_bqr_ald_all_reps[m_idx] <- rep_res$ess_tail_sigma_ald_bqr_ald; num_divergences_bqr_ald_all_reps[m_idx] <- if(is.na(rep_res$num_divergences_bqr_ald)) NA_real_ else as.numeric(rep_res$num_divergences_bqr_ald)
          # BSQR_G
          mse_beta_bsqr_g_all_reps[m_idx] <- rep_res$mse_beta_bsqr_g; mae_beta_bsqr_g_all_reps[m_idx] <- rep_res$mae_beta_bsqr_g; check_loss_test_bsqr_g_all_reps[m_idx] <- rep_res$check_loss_test_bsqr_g; ME_bsqr_g_all_reps[m_idx] <- rep_res$ME_bsqr_g; bias_beta_bsqr_g_list_all_reps[[m_idx]] <- rep_res$bias_beta_bsqr_g; coverage_beta_bsqr_g_list_all_reps[[m_idx]] <- rep_res$coverage_beta_bsqr_g; ci_width_beta_bsqr_g_list_all_reps[[m_idx]] <- rep_res$ci_width_beta_bsqr_g; fit_time_bsqr_g_all_reps[m_idx] <- rep_res$fit_time_bsqr_g; rhat_beta_max_bsqr_g_all_reps[m_idx] <- rep_res$rhat_beta_max_bsqr_g; ess_bulk_beta_min_bsqr_g_all_reps[m_idx] <- rep_res$ess_bulk_beta_min_bsqr_g; ess_tail_beta_min_bsqr_g_all_reps[m_idx] <- rep_res$ess_tail_beta_min_bsqr_g; rhat_theta_bsqr_g_all_reps[m_idx] <- rep_res$rhat_theta_bsqr_g; ess_bulk_theta_bsqr_g_all_reps[m_idx] <- rep_res$ess_bulk_theta_bsqr_g; ess_tail_theta_bsqr_g_all_reps[m_idx] <- rep_res$ess_tail_theta_bsqr_g; num_divergences_bsqr_g_all_reps[m_idx] <- if(is.na(rep_res$num_divergences_bsqr_g)) NA_real_ else as.numeric(rep_res$num_divergences_bsqr_g)
          # BSQR_U
          mse_beta_bsqr_u_all_reps[m_idx] <- rep_res$mse_beta_bsqr_u; mae_beta_bsqr_u_all_reps[m_idx] <- rep_res$mae_beta_bsqr_u; check_loss_test_bsqr_u_all_reps[m_idx] <- rep_res$check_loss_test_bsqr_u; ME_bsqr_u_all_reps[m_idx] <- rep_res$ME_bsqr_u; bias_beta_bsqr_u_list_all_reps[[m_idx]] <- rep_res$bias_beta_bsqr_u; coverage_beta_bsqr_u_list_all_reps[[m_idx]] <- rep_res$coverage_beta_bsqr_u; ci_width_beta_bsqr_u_list_all_reps[[m_idx]] <- rep_res$ci_width_beta_bsqr_u; fit_time_bsqr_u_all_reps[m_idx] <- rep_res$fit_time_bsqr_u; rhat_beta_max_bsqr_u_all_reps[m_idx] <- rep_res$rhat_beta_max_bsqr_u; ess_bulk_beta_min_bsqr_u_all_reps[m_idx] <- rep_res$ess_bulk_beta_min_bsqr_u; ess_tail_beta_min_bsqr_u_all_reps[m_idx] <- rep_res$ess_tail_beta_min_bsqr_u; rhat_theta_bsqr_u_all_reps[m_idx] <- rep_res$rhat_theta_bsqr_u; ess_bulk_theta_bsqr_u_all_reps[m_idx] <- rep_res$ess_bulk_theta_bsqr_u; ess_tail_theta_bsqr_u_all_reps[m_idx] <- rep_res$ess_tail_theta_bsqr_u; num_divergences_bsqr_u_all_reps[m_idx] <- if(is.na(rep_res$num_divergences_bsqr_u)) NA_real_ else as.numeric(rep_res$num_divergences_bsqr_u)
          # BSQR_E
          mse_beta_bsqr_e_all_reps[m_idx] <- rep_res$mse_beta_bsqr_e; mae_beta_bsqr_e_all_reps[m_idx] <- rep_res$mae_beta_bsqr_e; check_loss_test_bsqr_e_all_reps[m_idx] <- rep_res$check_loss_test_bsqr_e; ME_bsqr_e_all_reps[m_idx] <- rep_res$ME_bsqr_e; bias_beta_bsqr_e_list_all_reps[[m_idx]] <- rep_res$bias_beta_bsqr_e; coverage_beta_bsqr_e_list_all_reps[[m_idx]] <- rep_res$coverage_beta_bsqr_e; ci_width_beta_bsqr_e_list_all_reps[[m_idx]] <- rep_res$ci_width_beta_bsqr_e; fit_time_bsqr_e_all_reps[m_idx] <- rep_res$fit_time_bsqr_e; rhat_beta_max_bsqr_e_all_reps[m_idx] <- rep_res$rhat_beta_max_bsqr_e; ess_bulk_beta_min_bsqr_e_all_reps[m_idx] <- rep_res$ess_bulk_beta_min_bsqr_e; ess_tail_beta_min_bsqr_e_all_reps[m_idx] <- rep_res$ess_tail_beta_min_bsqr_e; rhat_theta_bsqr_e_all_reps[m_idx] <- rep_res$rhat_theta_bsqr_e; ess_bulk_theta_bsqr_e_all_reps[m_idx] <- rep_res$ess_bulk_theta_bsqr_e; ess_tail_theta_bsqr_e_all_reps[m_idx] <- rep_res$ess_tail_theta_bsqr_e; num_divergences_bsqr_e_all_reps[m_idx] <- if(is.na(rep_res$num_divergences_bsqr_e)) NA_real_ else as.numeric(rep_res$num_divergences_bsqr_e)
          # BSQR_T
          mse_beta_bsqr_t_all_reps[m_idx] <- rep_res$mse_beta_bsqr_t; mae_beta_bsqr_t_all_reps[m_idx] <- rep_res$mae_beta_bsqr_t; check_loss_test_bsqr_t_all_reps[m_idx] <- rep_res$check_loss_test_bsqr_t; ME_bsqr_t_all_reps[m_idx] <- rep_res$ME_bsqr_t; bias_beta_bsqr_t_list_all_reps[[m_idx]] <- rep_res$bias_beta_bsqr_t; coverage_beta_bsqr_t_list_all_reps[[m_idx]] <- rep_res$coverage_beta_bsqr_t; ci_width_beta_bsqr_t_list_all_reps[[m_idx]] <- rep_res$ci_width_beta_bsqr_t; fit_time_bsqr_t_all_reps[m_idx] <- rep_res$fit_time_bsqr_t; rhat_beta_max_bsqr_t_all_reps[m_idx] <- rep_res$rhat_beta_max_bsqr_t; ess_bulk_beta_min_bsqr_t_all_reps[m_idx] <- rep_res$ess_bulk_beta_min_bsqr_t; ess_tail_beta_min_bsqr_t_all_reps[m_idx] <- rep_res$ess_tail_beta_min_bsqr_t; rhat_theta_bsqr_t_all_reps[m_idx] <- rep_res$rhat_theta_bsqr_t; ess_bulk_theta_bsqr_t_all_reps[m_idx] <- rep_res$ess_bulk_theta_bsqr_t; ess_tail_theta_bsqr_t_all_reps[m_idx] <- rep_res$ess_tail_theta_bsqr_t; num_divergences_bsqr_t_all_reps[m_idx] <- if(is.na(rep_res$num_divergences_bsqr_t)) NA_real_ else as.numeric(rep_res$num_divergences_bsqr_t)
        } else {
          cat(paste0("@@@ MAIN PROCESS: Warning - Replication ", m_idx, " for config (Err=", current_error_spec$name_suffix, ", Scen=", scenario$name, ", Tau=", tau_val_current, ") returned NULL or invalid result. Filling with NAs.\n")); flush.console()
          bias_beta_rq_list_all_reps[[m_idx]] <- rep(NA_real_, p_current)
          bias_beta_bqr_ald_list_all_reps[[m_idx]] <- rep(NA_real_, p_current); coverage_beta_bqr_ald_list_all_reps[[m_idx]] <- rep(NA_real_, p_current); ci_width_beta_bqr_ald_list_all_reps[[m_idx]] <- rep(NA_real_, p_current)
          bias_beta_bsqr_g_list_all_reps[[m_idx]] <- rep(NA_real_, p_current); coverage_beta_bsqr_g_list_all_reps[[m_idx]] <- rep(NA_real_, p_current); ci_width_beta_bsqr_g_list_all_reps[[m_idx]] <- rep(NA_real_, p_current)
          bias_beta_bsqr_u_list_all_reps[[m_idx]] <- rep(NA_real_, p_current); coverage_beta_bsqr_u_list_all_reps[[m_idx]] <- rep(NA_real_, p_current); ci_width_beta_bsqr_u_list_all_reps[[m_idx]] <- rep(NA_real_, p_current)
          bias_beta_bsqr_e_list_all_reps[[m_idx]] <- rep(NA_real_, p_current); coverage_beta_bsqr_e_list_all_reps[[m_idx]] <- rep(NA_real_, p_current); ci_width_beta_bsqr_e_list_all_reps[[m_idx]] <- rep(NA_real_, p_current)
          bias_beta_bsqr_t_list_all_reps[[m_idx]] <- rep(NA_real_, p_current); coverage_beta_bsqr_t_list_all_reps[[m_idx]] <- rep(NA_real_, p_current); ci_width_beta_bsqr_t_list_all_reps[[m_idx]] <- rep(NA_real_, p_current)
        }
      }
      
      # Aggregate results for this configuration
      aggregate_list_of_vectors <- function(list_of_vectors, p_dim_expected) {
        valid_vectors <- Filter(function(x) { is.vector(x) && length(x) == p_dim_expected && !all(is.na(x)) }, list_of_vectors)
        if (length(valid_vectors) == 0) return(NA_real_)
        if(is.logical(valid_vectors[[1]])) valid_vectors <- lapply(valid_vectors, as.numeric)
        mean(colMeans(do.call(rbind, valid_vectors), na.rm = TRUE), na.rm = TRUE)
      }
      
      current_results_summary <- dplyr::bind_rows(
        # QR
        tibble::tibble(Error_Dist = current_error_spec$name_suffix, Scenario = scenario$name, Tau = tau_val_current, p = p_current, Method = "QR", Kernel = "N/A", MSE_beta = mean(mse_beta_rq_all_reps, na.rm=T), MAE_beta = mean(mae_beta_rq_all_reps, na.rm=T), ME_beta = mean(ME_rq_all_reps, na.rm=T), CheckLoss_test = mean(check_loss_test_rq_all_reps, na.rm=T), Bias_beta_avg = aggregate_list_of_vectors(bias_beta_rq_list_all_reps, p_current), Coverage_beta_avg = NA, CI_Width_beta_avg = NA, Fit_Time_avg_sec = NA, Rhat_beta_max = NA, ESS_bulk_beta_min = NA, ESS_tail_beta_min = NA, Rhat_theta = NA, ESS_bulk_theta = NA, ESS_tail_theta = NA, Num_Divergences_avg = NA),
        # BQR_ALD
        tibble::tibble(Error_Dist = current_error_spec$name_suffix, Scenario = scenario$name, Tau = tau_val_current, p = p_current, Method = "BQR", Kernel = "ALD", MSE_beta = mean(mse_beta_bqr_ald_all_reps, na.rm=T), MAE_beta = mean(mae_beta_bqr_ald_all_reps, na.rm=T), ME_beta = mean(ME_bqr_ald_all_reps, na.rm=T), CheckLoss_test = mean(check_loss_test_bqr_ald_all_reps, na.rm=T), Bias_beta_avg = aggregate_list_of_vectors(bias_beta_bqr_ald_list_all_reps, p_current), Coverage_beta_avg = aggregate_list_of_vectors(coverage_beta_bqr_ald_list_all_reps, p_current), CI_Width_beta_avg = aggregate_list_of_vectors(ci_width_beta_bqr_ald_list_all_reps, p_current), Fit_Time_avg_sec = mean(fit_time_bqr_ald_all_reps, na.rm=T), Rhat_beta_max = mean(rhat_beta_max_bqr_ald_all_reps, na.rm=T), ESS_bulk_beta_min = mean(ess_bulk_beta_min_bqr_ald_all_reps, na.rm=T), ESS_tail_beta_min = mean(ess_tail_beta_min_bqr_ald_all_reps, na.rm=T), Rhat_theta = mean(rhat_sigma_ald_bqr_ald_all_reps, na.rm=T), ESS_bulk_theta = mean(ess_bulk_sigma_ald_bqr_ald_all_reps, na.rm=T), ESS_tail_theta = mean(ess_tail_sigma_ald_bqr_ald_all_reps, na.rm=T), Num_Divergences_avg = mean(num_divergences_bqr_ald_all_reps, na.rm=T)),
        # BSQR_G
        tibble::tibble(Error_Dist = current_error_spec$name_suffix, Scenario = scenario$name, Tau = tau_val_current, p = p_current, Method = "BSQR", Kernel = "Gaussian", MSE_beta = mean(mse_beta_bsqr_g_all_reps, na.rm=T), MAE_beta = mean(mae_beta_bsqr_g_all_reps, na.rm=T), ME_beta = mean(ME_bsqr_g_all_reps, na.rm=T), CheckLoss_test = mean(check_loss_test_bsqr_g_all_reps, na.rm=T), Bias_beta_avg = aggregate_list_of_vectors(bias_beta_bsqr_g_list_all_reps, p_current), Coverage_beta_avg = aggregate_list_of_vectors(coverage_beta_bsqr_g_list_all_reps, p_current), CI_Width_beta_avg = aggregate_list_of_vectors(ci_width_beta_bsqr_g_list_all_reps, p_current), Fit_Time_avg_sec = mean(fit_time_bsqr_g_all_reps, na.rm=T), Rhat_beta_max = mean(rhat_beta_max_bsqr_g_all_reps, na.rm=T), ESS_bulk_beta_min = mean(ess_bulk_beta_min_bsqr_g_all_reps, na.rm=T), ESS_tail_beta_min = mean(ess_tail_beta_min_bsqr_g_all_reps, na.rm=T), Rhat_theta = mean(rhat_theta_bsqr_g_all_reps, na.rm=T), ESS_bulk_theta = mean(ess_bulk_theta_bsqr_g_all_reps, na.rm=T), ESS_tail_theta = mean(ess_tail_theta_bsqr_g_all_reps, na.rm=T), Num_Divergences_avg = mean(num_divergences_bsqr_g_all_reps, na.rm=T)),
        # BSQR_U
        tibble::tibble(Error_Dist = current_error_spec$name_suffix, Scenario = scenario$name, Tau = tau_val_current, p = p_current, Method = "BSQR", Kernel = "Uniform", MSE_beta = mean(mse_beta_bsqr_u_all_reps, na.rm=T), MAE_beta = mean(mae_beta_bsqr_u_all_reps, na.rm=T), ME_beta = mean(ME_bsqr_u_all_reps, na.rm=T), CheckLoss_test = mean(check_loss_test_bsqr_u_all_reps, na.rm=T), Bias_beta_avg = aggregate_list_of_vectors(bias_beta_bsqr_u_list_all_reps, p_current), Coverage_beta_avg = aggregate_list_of_vectors(coverage_beta_bsqr_u_list_all_reps, p_current), CI_Width_beta_avg = aggregate_list_of_vectors(ci_width_beta_bsqr_u_list_all_reps, p_current), Fit_Time_avg_sec = mean(fit_time_bsqr_u_all_reps, na.rm=T), Rhat_beta_max = mean(rhat_beta_max_bsqr_u_all_reps, na.rm=T), ESS_bulk_beta_min = mean(ess_bulk_beta_min_bsqr_u_all_reps, na.rm=T), ESS_tail_beta_min = mean(ess_tail_beta_min_bsqr_u_all_reps, na.rm=T), Rhat_theta = mean(rhat_theta_bsqr_u_all_reps, na.rm=T), ESS_bulk_theta = mean(ess_bulk_theta_bsqr_u_all_reps, na.rm=T), ESS_tail_theta = mean(ess_tail_theta_bsqr_u_all_reps, na.rm=T), Num_Divergences_avg = mean(num_divergences_bsqr_u_all_reps, na.rm=T)),
        # BSQR_E
        tibble::tibble(Error_Dist = current_error_spec$name_suffix, Scenario = scenario$name, Tau = tau_val_current, p = p_current, Method = "BSQR", Kernel = "Epanechnikov", MSE_beta = mean(mse_beta_bsqr_e_all_reps, na.rm=T), MAE_beta = mean(mae_beta_bsqr_e_all_reps, na.rm=T), ME_beta = mean(ME_bsqr_e_all_reps, na.rm=T), CheckLoss_test = mean(check_loss_test_bsqr_e_all_reps, na.rm=T), Bias_beta_avg = aggregate_list_of_vectors(bias_beta_bsqr_e_list_all_reps, p_current), Coverage_beta_avg = aggregate_list_of_vectors(coverage_beta_bsqr_e_list_all_reps, p_current), CI_Width_beta_avg = aggregate_list_of_vectors(ci_width_beta_bsqr_e_list_all_reps, p_current), Fit_Time_avg_sec = mean(fit_time_bsqr_e_all_reps, na.rm=T), Rhat_beta_max = mean(rhat_beta_max_bsqr_e_all_reps, na.rm=T), ESS_bulk_beta_min = mean(ess_bulk_beta_min_bsqr_e_all_reps, na.rm=T), ESS_tail_beta_min = mean(ess_tail_beta_min_bsqr_e_all_reps, na.rm=T), Rhat_theta = mean(rhat_theta_bsqr_e_all_reps, na.rm=T), ESS_bulk_theta = mean(ess_bulk_theta_bsqr_e_all_reps, na.rm=T), ESS_tail_theta = mean(ess_tail_theta_bsqr_e_all_reps, na.rm=T), Num_Divergences_avg = mean(num_divergences_bsqr_e_all_reps, na.rm=T)),
        # BSQR_T
        tibble::tibble(Error_Dist = current_error_spec$name_suffix, Scenario = scenario$name, Tau = tau_val_current, p = p_current, Method = "BSQR", Kernel = "Triangular", MSE_beta = mean(mse_beta_bsqr_t_all_reps, na.rm=T), MAE_beta = mean(mae_beta_bsqr_t_all_reps, na.rm=T), ME_beta = mean(ME_bsqr_t_all_reps, na.rm=T), CheckLoss_test = mean(check_loss_test_bsqr_t_all_reps, na.rm=T), Bias_beta_avg = aggregate_list_of_vectors(bias_beta_bsqr_t_list_all_reps, p_current), Coverage_beta_avg = aggregate_list_of_vectors(coverage_beta_bsqr_t_list_all_reps, p_current), CI_Width_beta_avg = aggregate_list_of_vectors(ci_width_beta_bsqr_t_list_all_reps, p_current), Fit_Time_avg_sec = mean(fit_time_bsqr_t_all_reps, na.rm=T), Rhat_beta_max = mean(rhat_beta_max_bsqr_t_all_reps, na.rm=T), ESS_bulk_beta_min = mean(ess_bulk_beta_min_bsqr_t_all_reps, na.rm=T), ESS_tail_beta_min = mean(ess_tail_beta_min_bsqr_t_all_reps, na.rm=T), Rhat_theta = mean(rhat_theta_bsqr_t_all_reps, na.rm=T), ESS_bulk_theta = mean(ess_bulk_theta_bsqr_t_all_reps, na.rm=T), ESS_tail_theta = mean(ess_tail_theta_bsqr_t_all_reps, na.rm=T), Num_Divergences_avg = mean(num_divergences_bsqr_t_all_reps, na.rm=T))
      )
      
      all_results_list[[length(all_results_list) + 1]] <- current_results_summary
      
      rep_end_time_tau <- Sys.time()
      cat(paste0("    Time for this configuration: ",
                 round(difftime(rep_end_time_tau, rep_start_time_tau, units = "secs"), 2), " seconds\n")); flush.console()
    }
  }
}

overall_end_time <- Sys.time()
cat(paste0("\n\nTotal simulation time: ", round(difftime(overall_end_time, overall_start_time, units = "hours"), 2), " hours\n")); flush.console()

future::plan(future::sequential)

#### --- 7. Consolidate and Save Results --- ####
if (length(all_results_list) > 0) {
  final_results_df <- dplyr::bind_rows(all_results_list) %>%
  select(Error_Dist, Scenario, Tau, Method, Kernel, p, MSE_beta, MAE_beta, ME_beta, CheckLoss_test, Bias_beta_avg, Coverage_beta_avg, CI_Width_beta_avg, Fit_Time_avg_sec, Rhat_beta_max, ESS_bulk_beta_min, ESS_tail_beta_min, Rhat_theta, ESS_bulk_theta, ESS_tail_theta, Num_Divergences_avg)
  cat("\n--- Simulation Results Summary ---\n"); flush.console()
  if(nrow(final_results_df) > 0){
    # 使用 cat() 直接输出文本，避免触发图形设备
    cat(knitr::kable(final_results_df, digits = 4, format = "pipe"), sep = "\n")
  } else {
    cat("No results to display in the summary table.\n"); flush.console()
  }
  
  
  results_filename_base <- paste0("BSQR_SimResults_AllKernels_", format(Sys.Date(), "%Y%m%d"), "_M", M, "_n", n_train)
  results_filename_suffix <- if (USE_TEST_MCMC_SETTINGS) "_TESTMCMC" else "_FULLMCMC"
  results_filename_cv <- if (USE_CV_FOR_H == 1) "_CV" else "_NoCV"
  results_filename <- paste0(results_filename_base, results_filename_suffix, results_filename_cv, ".csv")
  
  tryCatch({
    write.csv(final_results_df, results_filename, row.names = FALSE, na = "")
    cat(paste0("\nResults saved to: ", results_filename, "\n")); flush.console()
  }, error = function(e_csv){
    cat(paste0("\nError saving results to CSV: ", conditionMessage(e_csv), "\n"))
  })
} else {
  cat("\nNo results (all_results_list is empty).\n"); flush.console()
}
#### --- End of Script --- ####
