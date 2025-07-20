# Clean up environment
rm(list = ls()); gc()
set.seed(123)

#### --- 0. Global Parameters and Path Settings --- ####
RUN_TEST_MODE <- FALSE
jpm_csv_file_path <- "JPM.csv"
spx_csv_file_path <- "SPX.csv"
stan_file_bqr_ald <- "bqr_ald.stan"
stan_file_bsqr_uniform <- "bsqr_uniform_Z_robust.stan"
stan_file_bsqr_triangular <- "bsqr_triangular_Z_robust.stan"
window_size <- 252
step_size <- 21
capm_formula <- JPM_Return ~ GSPC_Return
tau_levels_to_run <- c(0.05, 0.95)
bqr_ald_mcmc_settings <- list(iter = 700 + 1500, warmup = 700, chains = 2, cores = 2, thin = 1, refresh = 0, control = list(adapt_delta = 0.95), brms_beta_prior_sd = sqrt(1000), brms_sigma_prior_scale = 2.5)
bsqr_main_mcmc_base_settings <- list(iter_warmup = 2000, iter_sampling = 2000, chains = 2, parallel_chains = 2, threads_per_chain = 1, adapt_delta = 0.99, max_treedepth = 15, refresh = 0, show_messages = FALSE, theta_gamma_shape_val = 2.0, theta_gamma_rate_val = 1.0, beta_loc_val_default = 0.0, beta_scale_val_default = 1.0, upper_bound_theta_val = 25.0, epsilon_theta_val = 1e-6)
bsqr_triangular_mcmc_settings <- c(bsqr_main_mcmc_base_settings, list(integration_rel_tol_val = 1e-7, k_asymptotic_switch = 5.0, use_asymptotic_approx = 0, theta_gamma_shape_val = 0.01, theta_gamma_rate_val = 0.01))
bsqr_uniform_mcmc_settings <- bsqr_main_mcmc_base_settings
cv_settings_base <- list(n_folds_cv = 5, h_grid_factors_cv = c(0.5, 0.75, 1.0, 1.5, 2.0), h_floor_value = 0.01, mcmc_params_cv_folds = list(iter_warmup = 800, iter_sampling = 600, chains = 2, parallel_chains = 2, thin = 1, refresh = 0, show_messages = FALSE, adapt_delta = 0.98, max_treedepth = 14), cv_seed_offset = 1000)
cv_settings_for_triangular <- c(cv_settings_base, list(mcmc_base_settings = bsqr_triangular_mcmc_settings))
cv_settings_for_uniform <- c(cv_settings_base, list(mcmc_base_settings = bsqr_uniform_mcmc_settings))


#### --- I. Load Necessary Packages --- ####
cat("--- Step I: Loading Required R Packages ---\n")
required_packages <- c("dplyr", "lubridate", "readr", "quantreg", "loo", "ggplot2", "tidyr", "caret", "cmdstanr", "posterior", "knitr", "e1071", "stringr", "colorspace")
for (pkg in required_packages) { if (!requireNamespace(pkg, quietly = TRUE)) { install.packages(pkg) }; library(pkg, character.only = TRUE) }
cat("Packages loaded.\n\n")


#### --- II. Data Preprocessing --- ####
cat("--- Step II: Data Preprocessing (Reading Raw Data) ---\n")
tryCatch({
  if (!file.exists(jpm_csv_file_path)) stop(paste("File not found:", jpm_csv_file_path))
  jpm_raw_df <- readr::read_csv(jpm_csv_file_path, col_types = readr::cols(Date = readr::col_date(format = "%Y-%m-%d"), `Adj Close` = readr::col_number()), show_col_types = FALSE)
  jpm_df <- jpm_raw_df %>% dplyr::select(Date, JPM_Adj_Close = `Adj Close`) %>% dplyr::arrange(Date)
  cat("JPM.csv read successfully.\n")
  if (!file.exists(spx_csv_file_path)) stop(paste("File not found:", spx_csv_file_path))
  spx_raw_df <- readr::read_csv(spx_csv_file_path, col_types = readr::cols(Date = readr::col_date(format = "%Y-%m-%d"), Close = readr::col_number()), show_col_types = FALSE)
  spx_df <- spx_raw_df %>% dplyr::select(Date, SPX_Close = Close) %>% dplyr::arrange(Date)
  cat("SPX.csv read successfully.\n")
  combined_prices_df <- dplyr::inner_join(jpm_df, spx_df, by = "Date")
  if(nrow(combined_prices_df) < 2) stop("Merged data has fewer than 2 common trading days.")
  returns_data_raw <- combined_prices_df %>% dplyr::mutate(JPM_Return = log(JPM_Adj_Close / dplyr::lag(JPM_Adj_Close)), GSPC_Return = log(SPX_Close / dplyr::lag(SPX_Close))) %>% dplyr::filter(!is.na(JPM_Return) & !is.na(GSPC_Return)) %>% dplyr::select(Date, JPM_Return, GSPC_Return) %>% dplyr::arrange(Date)
  if(nrow(returns_data_raw) == 0) stop("Unable to calculate valid return data.")
  cat("Raw log-returns calculated.\n\n")
}, error = function(e) { stop(paste("Fatal error during data preprocessing:", e$message)) })


cat("--- Step II-A: Generating descriptive statistics on the [Raw Data] ---\n")
if(nrow(returns_data_raw) > 1) {
  desc_stats_df <- data.frame( Statistic = c("Observations", "Mean", "Std. Dev.", "Skewness", "Kurtosis"), `JPMorgan Chase (JPM)` = c(nrow(returns_data_raw), mean(returns_data_raw$JPM_Return), sd(returns_data_raw$JPM_Return), e1071::skewness(returns_data_raw$JPM_Return), e1071::kurtosis(returns_data_raw$JPM_Return)), `S&P 500 Index (GSPC)` = c(nrow(returns_data_raw), mean(returns_data_raw$GSPC_Return), sd(returns_data_raw$GSPC_Return), e1071::skewness(returns_data_raw$GSPC_Return), e1071::kurtosis(returns_data_raw$GSPC_Return)), check.names = FALSE)
  cat("\n\n--- LaTeX Code for Descriptive Statistics (Table 1) ---\n")
  print(knitr::kable(desc_stats_df, format = "latex", booktabs = TRUE, digits = 4, caption = "Descriptive Statistics for Daily Log-Returns (Jan 2017 - Jan 2025).", label = "empirical_descriptive_stats_final_para"))
  cat("--- End of LaTeX Code ---\n\n")
}
cat("--- Step II-B: Scaling Data for Computational Stability ---\n")
returns_data <- returns_data_raw %>% mutate(JPM_Return = JPM_Return * 100, GSPC_Return = GSPC_Return * 100)
cat("Note: Returns data have been multiplied by 100 to improve MCMC numerical stability.\n'returns_data' dataframe has been created.\n--- Data preprocessing complete ---\n\n")


#### --- III. Define Helper Functions (Including Initialization and CI Logic) --- ####
cat("--- Step III: Defining Helper Functions ---\n")
rho_tau_loss <- function(u, tau_level) { ifelse(u >= 0, tau_level * u, (tau_level - 1) * u) }
calculate_h_silverman <- function(residuals_vec, num_obs_for_h, h_floor = 0.01) { if (length(residuals_vec) < 2 || num_obs_for_h < 2) return(h_floor); sigma_u_hat <- stats::sd(residuals_vec, na.rm = TRUE); if (is.na(sigma_u_hat) || sigma_u_hat == 0) { sigma_u_hat <- stats::mad(residuals_vec, center=0, na.rm = TRUE, constant = 1.4826) }; if (is.na(sigma_u_hat) || sigma_u_hat == 0) return(h_floor); h_calc <- 1.06 * sigma_u_hat * (num_obs_for_h ^ (-1/5)); return(max(h_calc, h_floor)) }
calculate_ci <- function(stan_fit_obj, variables, probs = c(0.025, 0.975)) {
  if (is.null(stan_fit_obj)) return(NULL)
  
  # Use the posterior package to safely extract draws
  draws_df <- tryCatch({
    posterior::as_draws_df(stan_fit_obj$draws(variables))
  }, error = function(e) {
    return(NULL)
  })
  
  if (is.null(draws_df) || nrow(draws_df) == 0) {
    return(NULL)
  }
  
  # Clean data and convert to long format
  draws_long <- draws_df %>%
    dplyr::select(-.chain, -.iteration, -.draw) %>%
    tidyr::pivot_longer(cols = everything(), names_to = "variable", values_to = "value")
  
  if (nrow(draws_long) == 0) {
    return(NULL)
  }
  
  # Calculate summary statistics, including quantiles
  final_summary <- draws_long %>%
    dplyr::group_by(variable) %>%
    dplyr::summarise(
      mean = mean(value, na.rm = TRUE),
      ess_bulk = posterior::ess_bulk(value),
      lower = stats::quantile(value, probs = probs[1], na.rm = TRUE),
      upper = stats::quantile(value, probs = probs[2], na.rm = TRUE),
      .groups = "drop"
    )
  
  return(final_summary)
}

# Data-driven initialization functions
provide_inits_bqr_ald <- function(chain_id, K_dim, seed_val, initial_beta, initial_sigma) {
  set.seed(chain_id + seed_val)
  beta_init <- if (!is.null(initial_beta) && length(initial_beta) == K_dim && all(is.finite(initial_beta))) initial_beta else rnorm(K_dim, 0, 0.1)
  sigma_init <- if (!is.null(initial_sigma) && is.finite(initial_sigma) && initial_sigma > 0) initial_sigma else runif(1, 0.01, 1)
  list(beta = beta_init, sigma = sigma_init)
}
provide_inits_for_stan_fit <- function(chain_id, K_dim_for_inits, epsilon_theta_for_inits, upper_bound_for_theta_for_inits, seed_val_for_init, initial_beta, initial_theta) {
  set.seed(seed_val_for_init + chain_id)
  beta_raw_init <- if (!is.null(initial_beta) && length(initial_beta) == K_dim_for_inits && all(is.finite(initial_beta))) initial_beta else rnorm(K_dim_for_inits, 0, 0.1)
  theta_init_candidate <- if(!is.null(initial_theta) && is.finite(initial_theta)) initial_theta else 2.0
  theta_init <- max(epsilon_theta_for_inits + 1e-5, min(upper_bound_for_theta_for_inits - 1e-5, theta_init_candidate))
  list(beta_raw = beta_raw_init, theta = theta_init)
}
cat("Helper functions defined.\n\n")


#### --- IV. Compile Stan Models --- ####
cat("--- Step IV: Compiling Stan Models (Suggestion: 'recompile = TRUE' if Stan files have changed) ---\n")
if (!file.exists(stan_file_bqr_ald)) stop("BQR-ALD Stan file not found!")
if (!file.exists(stan_file_bsqr_uniform)) stop("BSQR-Uniform Stan file not found!")
if (!file.exists(stan_file_bsqr_triangular)) stop("BSQR-Triangular Stan file not found!")
tryCatch({
  compiled_bqr_ald_model <- cmdstanr::cmdstan_model(stan_file = stan_file_bqr_ald, force_recompile = TRUE)
  cat("BQR-ALD Stan model compiled successfully.\n")
  compiled_bsqr_uniform_model <- cmdstanr::cmdstan_model(stan_file = stan_file_bsqr_uniform, force_recompile = TRUE)
  cat("BSQR-Uniform Stan model compiled successfully.\n")
  compiled_bsqr_triangular_model <- cmdstanr::cmdstan_model(stan_file = stan_file_bsqr_triangular, force_recompile = TRUE)
  cat("BSQR-Triangular Stan model compiled successfully.\n\n")
}, error = function(e) { stop(paste("Fatal error during Stan model compilation:", e$message)) })


#### --- V. Define Core Model Fitting Functions --- ####
cat("--- Step V: Defining Core Model Fitting Functions ---\n")
fit_bqr_ald_via_stan <- function(formula, data_df, tau_val, mcmc_params, compiled_model, seed_val, init_vals) {
  mf <- model.frame(formula, data_df); y <- model.response(mf); X <- model.matrix(formula, data_df); N <- nrow(X); K <- ncol(X)
  stan_data <- list(N = N, K = K, X = X, y = as.vector(y), tau = tau_val, beta_loc = 0.0, beta_scale = mcmc_params$brms_beta_prior_sd, sigma_cauchy_scale = mcmc_params$brms_sigma_prior_scale)
  init_fun <- function(chain_id) provide_inits_bqr_ald(chain_id, K, seed_val, init_vals$beta, init_vals$sigma)
  fit_start_time <- Sys.time()
  fit_obj <- compiled_model$sample(data = stan_data, seed = as.integer(seed_val), chains = mcmc_params$chains, parallel_chains = mcmc_params$cores, iter_warmup = mcmc_params$warmup, iter_sampling = mcmc_params$iter - mcmc_params$warmup, refresh = mcmc_params$refresh, adapt_delta = mcmc_params$control$adapt_delta, max_treedepth = 12, init = init_fun, show_messages = FALSE)
  fit_time <- as.numeric(difftime(Sys.time(), fit_start_time, units = "secs"))
  if(all(fit_obj$return_codes() == 0)) {
    num_divs <- sum(fit_obj$diagnostic_summary()$num_divergent, na.rm = TRUE)
    # [CORE MODIFICATION] Call the restored, reliable CI function
    summary_stats <- calculate_ci(fit_obj, variables = c("beta", "sigma"))
    if (!is.null(summary_stats)) {
      alpha_mean <- summary_stats$mean[summary_stats$variable == "beta[1]"]; beta_mean <- summary_stats$mean[summary_stats$variable == "beta[2]"]
      min_ess <- min(summary_stats$ess_bulk, na.rm = TRUE)
      beta_q_lower <- summary_stats$lower[summary_stats$variable == "beta[2]"]; beta_q_upper <- summary_stats$upper[summary_stats$variable == "beta[2]"]
      if(length(beta_q_lower) > 0 && length(beta_q_upper) > 0) {
        return(list(alpha = alpha_mean, beta = beta_mean, min_ess = min_ess, num_div = num_divs, time = fit_time, beta_lower = beta_q_lower, beta_upper = beta_q_upper, success = TRUE))
      }
    }
  }
  return(list(success = FALSE))
}
fit_bsqr_main_model_full <- function(stan_data_list, mcmc_settings_list, compiled_stan_model_obj, seed_val = 123, init_vals) {
  K_params <- stan_data_list$K
  init_fun_for_fit <- function(chain_id) { provide_inits_for_stan_fit(chain_id, K_params, mcmc_settings_list$epsilon_theta_val, mcmc_settings_list$upper_bound_theta_val, seed_val_for_init = seed_val, initial_beta = init_vals$beta, initial_theta = init_vals$theta) }
  sampling_start_time <- Sys.time()
  stan_fit_obj <- compiled_stan_model_obj$sample(data = stan_data_list, seed = as.integer(seed_val), chains = mcmc_settings_list$chains, parallel_chains = mcmc_settings_list$parallel_chains, iter_warmup = mcmc_settings_list$iter_warmup, iter_sampling = mcmc_settings_list$iter_sampling, refresh = mcmc_settings_list$refresh, adapt_delta = mcmc_settings_list$adapt_delta, max_treedepth = mcmc_settings_list$max_treedepth, init = init_fun_for_fit, show_messages = mcmc_settings_list$show_messages, diagnostics = c("divergences", "treedepth", "ebfmi"))
  fit_time_secs <- as.numeric(difftime(Sys.time(), sampling_start_time, units = "secs"))
  return(list(fit = stan_fit_obj, time_taken = fit_time_secs))
}
cat("Fitting functions defined.\n\n")


#### --- VI. Define Cross-Validation Function (No change needed) --- ####
cat("--- Step VI: Defining Cross-Validation Function for BSQR ---\n")
perform_cv_for_h_bsqr <- function(formula_cv, data_cv, tau_cv, cv_settings_list, compiled_stan_model_for_cv, kernel_type, seed_cv, log_prefix = "CV: ") {
  cat(paste0(log_prefix, "Starting CV for h (", kernel_type, " Kernel, tau=", tau_cv, "). Random Seed: ", seed_cv, "\n"))
  mf_cv <- model.frame(formula_cv, data = data_cv); y_cv_full <- model.response(mf_cv); X_cv_full <- model.matrix(formula_cv, data = data_cv); K_params <- ncol(X_cv_full)
  initial_residuals_for_h_cv <- quantreg::rq(formula_cv, tau=tau_cv, data=data_cv, method="br")$residuals; h_silverman_base_cv <- calculate_h_silverman(initial_residuals_for_h_cv, nrow(X_cv_full), h_floor = cv_settings_list$h_floor_value); current_h_grid <- sort(unique(pmax(cv_settings_list$h_floor_value, h_silverman_base_cv * cv_settings_list$h_grid_factors_cv)))
  cat(paste0(log_prefix, "Base h value=", round(h_silverman_base_cv, 5), ", CV h grid: [", paste(round(current_h_grid, 4), collapse=", "), "]\n"))
  if (is.null(compiled_stan_model_for_cv)) { fallback_idx <- ceiling(length(current_h_grid) / 2); return(current_h_grid[fallback_idx]) }
  set.seed(seed_cv); folds_list <- caret::createFolds(y_cv_full, k = cv_settings_list$n_folds_cv, list = TRUE, returnTrain = FALSE); avg_check_losses_cv <- numeric(length(current_h_grid)); names(avg_check_losses_cv) <- as.character(current_h_grid)
  for (i_h in seq_along(current_h_grid)) {
    h_candidate_cv <- current_h_grid[i_h]; fold_check_losses <- numeric(cv_settings_list$n_folds_cv)
    for (k_fold in 1:cv_settings_list$n_folds_cv) {
      val_indices <- folds_list[[k_fold]]; train_indices <- setdiff(1:length(y_cv_full), val_indices); data_train_fold <- data_cv[train_indices, ]; data_val_fold <- data_cv[val_indices, ]; X_train_mat_fold <- model.matrix(formula_cv, data = data_train_fold); y_train_vec_fold <- model.response(model.frame(formula_cv, data = data_train_fold)); fold_check_losses[k_fold] <- Inf; beta_est_fold <- rep(NA, K_params)
      tryCatch({
        mcmc_params_complete_for_cv <- cv_settings_list$mcmc_base_settings; cv_specific_overrides <- cv_settings_list$mcmc_params_cv_folds; for(param_name in names(cv_specific_overrides)){ mcmc_params_complete_for_cv[[param_name]] <- cv_specific_overrides[[param_name]] }
        stan_data_fold_base <- list(N_train_obs = nrow(X_train_mat_fold), K = K_params, X_train = X_train_mat_fold, y_train = as.vector(y_train_vec_fold), tau = tau_cv, h = h_candidate_cv, upper_bound_for_theta = mcmc_params_complete_for_cv$upper_bound_theta_val, epsilon_theta = mcmc_params_complete_for_cv$epsilon_theta_val, beta_location = rep(mcmc_params_complete_for_cv$beta_loc_val_default, K_params), beta_scale = rep(mcmc_params_complete_for_cv$beta_scale_val_default, K_params))
        if (kernel_type == "uniform") { stan_data_fold <- c(stan_data_fold_base, list(gamma_shape = mcmc_params_complete_for_cv$theta_gamma_shape_val, theta_prior_rate_val = mcmc_params_complete_for_cv$theta_gamma_rate_val))
        } else if (kernel_type == "triangular") { base_settings_for_tri_cv <- cv_settings_list$mcmc_base_settings; stan_data_fold <- c(stan_data_fold_base, list(gamma_shape = base_settings_for_tri_cv$theta_gamma_shape_val, gamma_rate = base_settings_for_tri_cv$theta_gamma_rate_val, Z_rel_tol = base_settings_for_tri_cv$integration_rel_tol_val, K_ASYMPTOTIC_SWITCH_STD_DEVS = base_settings_for_tri_cv$k_asymptotic_switch, USE_Z_ASYMPTOTIC_APPROX = base_settings_for_tri_cv$use_asymptotic_approx ))
        } else { stop("Unknown kernel type provided to CV function.") }
        
        cv_rq_fit <- quantreg::rq(formula_cv, tau = tau_cv, data = data_train_fold, method = "br")
        cv_init_beta <- coef(cv_rq_fit)
        cv_init_h <- calculate_h_silverman(residuals(cv_rq_fit), nrow(data_train_fold))
        cv_init_theta <- if(cv_init_h > 0) 1/cv_init_h else NULL
        cv_init_vals <- list(beta = cv_init_beta, theta = cv_init_theta)
        
        fit_bundle_cv <- fit_bsqr_main_model_full(stan_data_list = stan_data_fold, mcmc_settings_list = mcmc_params_complete_for_cv, compiled_stan_model_obj = compiled_stan_model_for_cv, seed_val = as.integer(seed_cv + i_h*100 + k_fold*10), init_vals = cv_init_vals)
        if (!is.null(fit_bundle_cv$fit) && all(fit_bundle_cv$fit$return_codes() == 0)) {
          summary_cv <- calculate_ci(fit_bundle_cv$fit, variables = c("beta"))
          if (!is.null(summary_cv) && nrow(summary_cv) == K_params) { beta_est_fold <- summary_cv$mean }
        }
      }, error = function(e_cv) {})
      if (!any(is.na(beta_est_fold))) { X_val_mat <- model.matrix(formula_cv, data_val_fold); y_val_vec <- model.response(model.frame(formula_cv, data_val_fold)); val_preds_cv <- X_val_mat %*% beta_est_fold; fold_check_losses[k_fold] <- mean(rho_tau_loss(y_val_vec - val_preds_cv, tau_cv), na.rm = TRUE) }
    }
    avg_check_losses_cv[i_h] <- mean(fold_check_losses[is.finite(fold_check_losses)], na.rm = TRUE); if(is.nan(avg_check_losses_cv[i_h])) avg_check_losses_cv[i_h] <- Inf
  }
  best_h_index <- which.min(avg_check_losses_cv)
  if (length(best_h_index) == 0 || !is.finite(avg_check_losses_cv[best_h_index])) { fallback_idx <- ceiling(length(current_h_grid) / 2); optimal_h_cv <- current_h_grid[fallback_idx]; cat(paste0(log_prefix, kernel_type, " CV failed to find a valid optimal h, using fallback value: ", round(optimal_h_cv, 5), "\n"))
  } else { optimal_h_cv <- current_h_grid[best_h_index[1]]; cat(paste0(log_prefix, kernel_type, " CV selected optimal h: ", round(optimal_h_cv, 5), "\n")) }
  return(optimal_h_cv)
}
cat("CV function defined.\n\n")


#### --- VII. Rolling-Window Analysis --- ####
cat("--- Step VII: Starting Rolling-Window Analysis ---\n")
master_results_beta <- list()
master_results_forecasts <- list()
master_results_mcmc <- list()
for (current_tau in tau_levels_to_run) {
  cat(paste0("\n\n======================================================\n"))
  cat(paste0("   Starting analysis for quantile TAU = ", current_tau, "...\n"))
  cat(paste0("======================================================\n\n"))
  n_total_obs <- nrow(returns_data)
  start_indices <- seq(from = 1, to = n_total_obs - window_size + 1, by = step_size)
  num_windows_to_run <- length(start_indices)
  cat("Total number of windows to process:", num_windows_to_run, "\n")
  
  if (exists("RUN_TEST_MODE") && RUN_TEST_MODE == TRUE) {
    num_test_windows <- 5
    if (num_windows_to_run >= num_test_windows) {
      start_indices <- head(start_indices, num_test_windows)
      num_windows_to_run <- length(start_indices)
      cat(paste0("\n\n>>>>>> TEST MODE ACTIVATED <<<<<<\nProcessing only the first ", num_windows_to_run, " windows.\nTo run the full analysis, set RUN_TEST_MODE to FALSE at the top of the script.\n\n"))
    } else {
      cat(paste0("\n\n>>>>>> TEST MODE WARNING <<<<<<\nRequested ", num_test_windows, " test windows, but only ", num_windows_to_run, " are available. Running all available windows.\n\n"))
    }
  }
  
  if (num_windows_to_run < 1) { warning(paste("Insufficient data for a single rolling window (tau =", current_tau, "). Skipping...")); next }
  results_list_beta_current_tau <- list()
  results_list_forecasts_current_tau <- list()
  results_list_mcmc_current_tau <- list()
  pb <- txtProgressBar(min = 0, max = num_windows_to_run, style = 3, char = paste0("tau=", current_tau, ":="))
  for (i in seq_along(start_indices)) {
    i_start_index <- start_indices[i]
    current_loop_master_seed <- 12345 + i + (which(tau_levels_to_run == current_tau) * 10000)
    window_end_index <- i_start_index + window_size - 1
    current_window_df <- returns_data[i_start_index:window_end_index, ]
    window_end_dt <- current_window_df$Date[window_size]
    
    initial_rq_fit <- NULL
    tryCatch({
      initial_rq_fit <- quantreg::rq(capm_formula, tau = current_tau, data = current_window_df, method = "br")
    }, error = function(e){ cat("\n quantreg::rq() in window ", i, " failed. Will use default inits.\n")})
    
    initial_beta_est <- if(!is.null(initial_rq_fit)) coef(initial_rq_fit) else NULL
    initial_residuals <- if(!is.null(initial_rq_fit)) residuals(initial_rq_fit) else NULL
    initial_sigma_est <- if(!is.null(initial_residuals)) mad(initial_residuals, constant = 1) else NULL
    initial_h_est <- if(!is.null(initial_residuals)) calculate_h_silverman(initial_residuals, nrow(current_window_df)) else NULL
    initial_theta_est <- if(!is.null(initial_h_est) && initial_h_est > 0) 1/initial_h_est else NULL
    
    init_vals_bqr_ald <- list(beta = initial_beta_est, sigma = initial_sigma_est)
    init_vals_bsqr <- list(beta = initial_beta_est, theta = initial_theta_est)
    
    # --- Model 1: BQR-ALD ---
    alpha_bqr_ald <- NA; beta_bqr_ald <- NA;
    tryCatch({
      bqr_ald_results <- fit_bqr_ald_via_stan(formula = capm_formula, data_df = current_window_df, tau_val = current_tau, mcmc_params = bqr_ald_mcmc_settings, compiled_model = compiled_bqr_ald_model, seed_val = as.integer(current_loop_master_seed + 100), init_vals = init_vals_bqr_ald)
      if(bqr_ald_results$success) {
        alpha_bqr_ald <- bqr_ald_results$alpha; beta_bqr_ald <- bqr_ald_results$beta
        results_list_beta_current_tau[[length(results_list_beta_current_tau) + 1]] <- data.frame(Date = window_end_dt, Model = "BQR-ALD", Tau_Level = current_tau, Beta_Mean = bqr_ald_results$beta, Beta_LowerCI = bqr_ald_results$beta_lower, Beta_UpperCI = bqr_ald_results$beta_upper)
        results_list_mcmc_current_tau[[length(results_list_mcmc_current_tau) + 1]] <- data.frame(Date = window_end_dt, Model = "BQR-ALD", Tau_Level = current_tau, MinESS = bqr_ald_results$min_ess, Time = bqr_ald_results$time, NumDiv = bqr_ald_results$num_div, h_Selected = NA)
      } else { cat("\nBQR-ALD failed to extract results in window ", i, ", skipping this model.\n") }
    }, error = function(e) { cat("\nBQR-ALD failed in window ", i, ": ", e$message, "\n") })
    
    # --- Model 2: BSQR-Uniform ---
    alpha_bsqr_u <- NA; beta_bsqr_u <- NA
    tryCatch({
      h_uniform <- perform_cv_for_h_bsqr(capm_formula, current_window_df, current_tau, cv_settings_for_uniform, compiled_bsqr_uniform_model, "uniform", as.integer(current_loop_master_seed + 2000), log_prefix="    ")
      mf_main_u <- model.frame(capm_formula, current_window_df); X_main_u <- model.matrix(capm_formula, current_window_df); y_main_u <- model.response(mf_main_u); K_u <- ncol(X_main_u)
      stan_data_u <- list(N_train_obs = nrow(current_window_df), K = K_u, X_train = X_main_u, y_train = y_main_u, tau = current_tau, h = h_uniform, gamma_shape = bsqr_uniform_mcmc_settings$theta_gamma_shape_val, theta_prior_rate_val = bsqr_uniform_mcmc_settings$theta_gamma_rate_val, upper_bound_for_theta = bsqr_uniform_mcmc_settings$upper_bound_theta_val, epsilon_theta = bsqr_uniform_mcmc_settings$epsilon_theta_val, beta_location = rep(bsqr_uniform_mcmc_settings$beta_loc_val_default, K_u), beta_scale = rep(bsqr_uniform_mcmc_settings$beta_scale_val_default, K_u))
      main_fit_bundle_u <- fit_bsqr_main_model_full(stan_data_u, bsqr_uniform_mcmc_settings, compiled_bsqr_uniform_model, as.integer(current_loop_master_seed + 200), init_vals = init_vals_bsqr)
      if (!is.null(main_fit_bundle_u$fit) && all(main_fit_bundle_u$fit$return_codes() == 0)) {
        summary_u <- calculate_ci(main_fit_bundle_u$fit, variables = c("beta", "theta"))
        if (!is.null(summary_u)) {
          alpha_bsqr_u <- summary_u$mean[summary_u$variable == "beta[1]"]; beta_bsqr_u <- summary_u$mean[summary_u$variable == "beta[2]"]
          beta_lower_u <- summary_u$lower[summary_u$variable == "beta[2]"]; beta_upper_u <- summary_u$upper[summary_u$variable == "beta[2]"]
          if (length(alpha_bsqr_u) > 0 && length(beta_bsqr_u) > 0 && length(beta_lower_u) > 0 && length(beta_upper_u) > 0) {
            min_ess_u <- min(summary_u$ess_bulk, na.rm = TRUE); num_div_u <- sum(main_fit_bundle_u$fit$diagnostic_summary()$num_divergent, na.rm = TRUE)
            results_list_beta_current_tau[[length(results_list_beta_current_tau) + 1]] <- data.frame(Date = window_end_dt, Model = "BSQR-Uniform", Tau_Level = current_tau, Beta_Mean = beta_bsqr_u, Beta_LowerCI = beta_lower_u, Beta_UpperCI = beta_upper_u)
            results_list_mcmc_current_tau[[length(results_list_mcmc_current_tau) + 1]] <- data.frame(Date = window_end_dt, Model = "BSQR-Uniform", Tau_Level = current_tau, MinESS = min_ess_u, Time = main_fit_bundle_u$time_taken, NumDiv = num_div_u, h_Selected = h_uniform)
          } else { cat("\nBSQR-Uniform failed to extract results in window ", i, ", skipping this model.\n") }
        } else { cat("\nBSQR-Uniform CI calculation failed in window ", i, ", skipping this model.\n") }
      } else { cat("\nBSQR-Uniform sampling failed in window ", i, ", skipping this model.\n") }
    }, error = function(e) { cat("\nBSQR-Uniform failed in window ", i, ": ", e$message, "\n") })
    
    # --- Model 3: BSQR-Triangular ---
    alpha_bsqr_t <- NA; beta_bsqr_t <- NA
    tryCatch({
      h_triangular <- perform_cv_for_h_bsqr(capm_formula, current_window_df, current_tau, cv_settings_for_triangular, compiled_bsqr_triangular_model, "triangular", as.integer(current_loop_master_seed + 3000), log_prefix="    ")
      mf_main_t <- model.frame(capm_formula, current_window_df); X_main_t <- model.matrix(capm_formula, current_window_df); y_main_t <- model.response(mf_main_t); K_t <- ncol(X_main_t)
      stan_data_t <- list(N_train_obs = nrow(current_window_df), K = K_t, X_train = X_main_t, y_train = y_main_t, tau = current_tau, h = h_triangular, gamma_shape = bsqr_triangular_mcmc_settings$theta_gamma_shape_val, gamma_rate = bsqr_triangular_mcmc_settings$theta_gamma_rate_val, upper_bound_for_theta = bsqr_triangular_mcmc_settings$upper_bound_theta_val, epsilon_theta = bsqr_triangular_mcmc_settings$epsilon_theta_val, beta_location = rep(bsqr_triangular_mcmc_settings$beta_loc_val_default, K_t), beta_scale = rep(bsqr_triangular_mcmc_settings$beta_scale_val_default, K_t), Z_rel_tol = bsqr_triangular_mcmc_settings$integration_rel_tol_val, K_ASYMPTOTIC_SWITCH_STD_DEVS = bsqr_triangular_mcmc_settings$k_asymptotic_switch, USE_Z_ASYMPTOTIC_APPROX = bsqr_triangular_mcmc_settings$use_asymptotic_approx )
      main_fit_bundle_t <- fit_bsqr_main_model_full(stan_data_t, bsqr_triangular_mcmc_settings, compiled_bsqr_triangular_model, as.integer(current_loop_master_seed + 300), init_vals = init_vals_bsqr)
      if (!is.null(main_fit_bundle_t$fit) && all(main_fit_bundle_t$fit$return_codes() == 0)) {
        summary_t <- calculate_ci(main_fit_bundle_t$fit, variables = c("beta", "theta"))
        if (!is.null(summary_t)) {
          alpha_bsqr_t <- summary_t$mean[summary_t$variable == "beta[1]"]; beta_bsqr_t <- summary_t$mean[summary_t$variable == "beta[2]"]
          beta_lower_t <- summary_t$lower[summary_t$variable == "beta[2]"]; beta_upper_t <- summary_t$upper[summary_t$variable == "beta[2]"]
          if (length(alpha_bsqr_t) > 0 && length(beta_bsqr_t) > 0 && length(beta_lower_t) > 0 && length(beta_upper_t) > 0) {
            min_ess_t <- min(summary_t$ess_bulk, na.rm = TRUE); num_div_t <- sum(main_fit_bundle_t$fit$diagnostic_summary()$num_divergent, na.rm = TRUE)
            results_list_beta_current_tau[[length(results_list_beta_current_tau) + 1]] <- data.frame(Date = window_end_dt, Model = "BSQR-Triangular", Tau_Level = current_tau, Beta_Mean = beta_bsqr_t, Beta_LowerCI = beta_lower_t, Beta_UpperCI = beta_upper_t)
            results_list_mcmc_current_tau[[length(results_list_mcmc_current_tau) + 1]] <- data.frame(Date = window_end_dt, Model = "BSQR-Triangular", Tau_Level = current_tau, MinESS = min_ess_t, Time = main_fit_bundle_t$time_taken, NumDiv = num_div_t, h_Selected = h_triangular)
          } else { cat("\nBSQR-Triangular failed to extract results in window ", i, ", skipping this model.\n") }
        } else { cat("\nBSQR-Triangular CI calculation failed in window ", i, ", skipping this model.\n") }
      } else { cat("\nBSQR-Triangular sampling failed in window ", i, ", skipping this model.\n") }
    }, error = function(e) { cat("\nBSQR-Triangular failed in window ", i, ": ", e$message, "\n") })
    
    # --- Out-of-Sample Forecasting ---
    if (window_end_index < n_total_obs) {
      forecast_data_pt_df <- returns_data[window_end_index + 1, ]; actual_next_jpm_ret <- forecast_data_pt_df$JPM_Return; market_ret_next_day <- forecast_data_pt_df$GSPC_Return
      cl_bqr_ald <- if (!is.na(alpha_bqr_ald) && length(alpha_bqr_ald)==1) rho_tau_loss(actual_next_jpm_ret - (alpha_bqr_ald + beta_bqr_ald * market_ret_next_day), current_tau) else NA
      cl_bsqr_u <- if (!is.na(alpha_bsqr_u) && length(alpha_bsqr_u)==1) rho_tau_loss(actual_next_jpm_ret - (alpha_bsqr_u + beta_bsqr_u * market_ret_next_day), current_tau) else NA
      cl_bsqr_t <- if (!is.na(alpha_bsqr_t) && length(alpha_bsqr_t)==1) rho_tau_loss(actual_next_jpm_ret - (alpha_bsqr_t + beta_bsqr_t * market_ret_next_day), current_tau) else NA
      results_list_forecasts_current_tau[[length(results_list_forecasts_current_tau) + 1]] <- data.frame(Date = forecast_data_pt_df$Date, Tau_Level = current_tau, CheckLoss_BQR_ALD = cl_bqr_ald, CheckLoss_BSQR_Uniform = cl_bsqr_u, CheckLoss_BSQR_Triangular = cl_bsqr_t)
    }
    setTxtProgressBar(pb, i)
  }
  close(pb)
  if (length(results_list_beta_current_tau) > 0) master_results_beta[[as.character(current_tau)]] <- bind_rows(results_list_beta_current_tau)
  if (length(results_list_forecasts_current_tau) > 0) master_results_forecasts[[as.character(current_tau)]] <- bind_rows(results_list_forecasts_current_tau)
  if (length(results_list_mcmc_current_tau) > 0) master_results_mcmc[[as.character(current_tau)]] <- bind_rows(results_list_mcmc_current_tau)
  cat(paste0("\n--- Rolling-window analysis for quantile TAU = ", current_tau, " complete ---\n\n"))
}
df_beta_all_models <- bind_rows(master_results_beta)
df_forecasts_all_models <- bind_rows(master_results_forecasts)
df_mcmc_all_models <- bind_rows(master_results_mcmc)

#### --- VIII. Generate Figures and Tables --- ####
cat("--- Step VIII: Generating Figures and Tables (Results are on the original scale) ---\n")
for (tau_to_process in tau_levels_to_run) {
  cat(paste0("\n\n--- Starting to generate results for TAU = ", tau_to_process, " ---\n"))
  df_beta_subset <- df_beta_all_models %>% filter(as.numeric(Tau_Level) == tau_to_process)
  if (nrow(df_beta_subset) > 0 && sum(!is.na(df_beta_subset$Beta_Mean)) > 0) {
    df_beta_subset$Date <- as.Date(df_beta_subset$Date)
    
    color_palette_lines <- c("BQR-ALD" = "gray20", "BSQR-Uniform" = "#0072B2", "BSQR-Triangular" = "#D55E00")
    eps_fill_palette <- colorspace::lighten(color_palette_lines, amount = 0.6)
    alpha_palette_for_png <- c("BQR-ALD" = 0.15, "BSQR-Uniform" = 0.15, "BSQR-Triangular" = 0.20)
    linetype_palette <- c("BQR-ALD" = "dashed", "BSQR-Uniform" = "solid", "BSQR-Triangular" = "solid")
    linewidth_palette <- c("BQR-ALD" = 0.8, "BSQR-Uniform" = 1.0, "BSQR-Triangular" = 1.2)
    
    beta_type_text <- if (tau_to_process == 0.05) "Downside Beta" else "Upside Beta"; plot_title_expr <- bquote("Evolution of JPM's Dynamic " * .(beta_type_text) * " (" * tau * " = " * .(tau_to_process) * ")"); plot_y_axis_expr <- bquote(.(beta_type_text) * ", " * beta(tau)); figure_filename_base <- paste0("Figure_Beta_tau_", stringr::str_replace(as.character(tau_to_process), "\\.", "_")); plot_subtitle_text_raw <- paste0("Comparison of the benchmark (BQR-ALD) with BSQR methods (Uniform and Triangular kernels).\nAnalysis based on a ", window_size, "-day rolling window. Shaded areas represent 95% credible intervals.")
    common_theme <- theme_bw(base_size = 14) + theme(plot.title = element_text(face = "bold", size = rel(1.2), hjust = 0.5), plot.subtitle = element_text(size = rel(0.9), hjust = 0.5, color = "gray20", lineheight = 1.1), axis.title = element_text(face = "bold"), legend.position = "top", legend.title = element_blank(), legend.background = element_rect(fill = "white", color = "gray80"), legend.key.width = unit(1.5, "cm"), panel.grid.minor = element_blank(), panel.grid.major = element_line(color = "gray85", linetype = "dashed"), panel.border = element_rect(color = "black", fill = NA, linewidth = 0.7))
    y_max_limit <- if(all(is.na(df_beta_subset$Beta_UpperCI))) 1 else max(df_beta_subset$Beta_UpperCI, na.rm = TRUE)
    common_layers <- list(geom_vline(xintercept = as.Date("2020-03-11"), linetype = "dotted", color = "firebrick", linewidth = 0.8), annotate("text", x = as.Date("2020-03-11") + days(15), y = y_max_limit * 0.98, label = "COVID-19\nPandemic Declared", hjust = 0, color = "firebrick", size = 3.5, lineheight = 0.9), scale_color_manual(values = color_palette_lines), scale_linetype_manual(values = linetype_palette), scale_linewidth_manual(values = linewidth_palette), scale_x_date(date_breaks = "2 years", date_labels = "%Y"), labs(title = plot_title_expr, subtitle = plot_subtitle_text_raw, x = "End Date of Rolling Window", y = plot_y_axis_expr), common_theme)
    
    plot_for_png <- ggplot(df_beta_subset, aes(x = Date, color = Model, linetype = Model)) + geom_ribbon(aes(ymin = Beta_LowerCI, ymax = Beta_UpperCI, fill = Model, alpha = Model), show.legend = FALSE, na.rm = TRUE, linetype = 0) + geom_line(aes(y = Beta_Mean, linewidth = Model), na.rm = TRUE) + scale_fill_manual(values = color_palette_lines) + scale_alpha_manual(values = alpha_palette_for_png) + common_layers
    ggsave(paste0(figure_filename_base, ".png"), plot = plot_for_png, width = 11, height = 7, dpi = 300, device = "png"); cat(paste0("\nRolling Beta plot (tau=", tau_to_process, ") saved as ", figure_filename_base, ".png\n"))

    plot_for_eps <- ggplot(df_beta_subset, aes(x = Date, color = Model, linetype = Model)) + geom_ribbon(aes(ymin = Beta_LowerCI, ymax = Beta_UpperCI, fill = Model), show.legend = FALSE, na.rm = TRUE, linetype = 0) + geom_line(aes(y = Beta_Mean, linewidth = Model), na.rm = TRUE) + scale_fill_manual(values = eps_fill_palette) + common_layers
    ggsave(paste0(figure_filename_base, ".eps"), plot = plot_for_eps, width = 11, height = 7, device = "eps"); cat(paste0("Rolling Beta plot (tau=", tau_to_process, ") saved as ", figure_filename_base, ".eps\n"))
    
  } else { 
    cat(paste0("\nInsufficient data or all model beta results are NA for tau=", tau_to_process, "; cannot generate rolling Beta plot.\n")) 
  }
}

cat("\n\n--- Generating final summary tables matching the manuscript ---\n")
if (nrow(df_forecasts_all_models) > 0) {
  forecast_summary_for_latex <- df_forecasts_all_models %>% pivot_longer(cols = starts_with("CheckLoss_"), names_to = "Model", values_to = "CheckLoss") %>% mutate(Model = stringr::str_remove(Model, "CheckLoss_")) %>% group_by(Model, Tau_Level) %>% summarise(Average_Check_Loss = mean(CheckLoss, na.rm = TRUE) / 100, .groups = 'drop') %>% pivot_wider(names_from = Tau_Level, values_from = Average_Check_Loss, names_prefix = "tau_") %>% arrange(factor(Model, levels = c("BQR-ALD", "BSQR-Uniform", "BSQR-Triangular")))
  cat("\n\n--- LaTeX Code for Forecasting Performance Table (Table 2) ---\n"); print(knitr::kable(forecast_summary_for_latex, format = "latex", booktabs = TRUE, digits = 6, caption = "Out-of-Sample Forecasting Performance for Downside and Upside Quantiles.", label = "forecast_results_asymmetric_detailed", col.names = c("Model", "Downside ($\\tau=0.05$)", "Upside ($\\tau=0.95$)"), escape = FALSE)); cat("--- End of LaTeX Code ---\n")
}
if (nrow(df_mcmc_all_models) > 0) {
  mcmc_summary_for_latex <- df_mcmc_all_models %>% group_by(Model) %>% summarise(`Avg. Min. ESS` = mean(MinESS, na.rm = TRUE), `Avg. Time (s)` = mean(Time, na.rm = TRUE), `Avg. Divergences` = mean(NumDiv, na.rm = TRUE), `Avg. Selected $h$` = if (first(Model) == "BQR-ALD") NA_real_ else mean(h_Selected, na.rm = TRUE)) %>% arrange(factor(Model, levels = c("BQR-ALD", "BSQR-Uniform", "BSQR-Triangular")))
  cat("\n\n--- LaTeX Code for MCMC Sampling Efficiency Table (Table 3) ---\n"); print(knitr::kable(mcmc_summary_for_latex, format = "latex", booktabs = TRUE, digits = 2, caption = "MCMC Sampler Efficiency in Rolling-Window Analysis.", label = "mcmc_efficiency_asymmetric_detailed", col.names = c("Model", "Avg. Min. ESS", "Avg. Time (s)", "Avg. Divergences", "Avg. Selected $h$"))); cat("--- End of LaTeX Code ---\n")
}

#### --- IX. Data analysis in figures --- ####
if (!require(dplyr)) { install.packages("dplyr"); library(dplyr) }
if (!require(lubridate)) { install.packages("lubridate"); library(lubridate) }

# Define the model we will use for the analysis (BSQR-Uniform is a robust choice)
MODEL_TO_ANALYZE <- "BSQR-Uniform"

# --- Define Time Periods ---
PRE_CRISIS_START <- as.Date("2019-01-01")
PRE_CRISIS_END   <- as.Date("2019-12-31")
CRISIS_START     <- as.Date("2020-03-01")
CRISIS_END       <- as.Date("2021-12-31")
POST_CRISIS_START<- as.Date("2022-01-01")
POST_CRISIS_END  <- max(df_beta_all_models$Date) 

# --- Function to extract detailed stats for a given tau ---
get_detailed_stats <- function(tau_val) {
  
  # 1. Pre-Crisis Period
  pre_crisis_data <- df_beta_all_models %>%
    filter(Model == MODEL_TO_ANALYZE, 
           Tau_Level == tau_val,
           Date >= PRE_CRISIS_START, Date <= PRE_CRISIS_END)
  
  pre_crisis_mean <- mean(pre_crisis_data$Beta_Mean, na.rm = TRUE)
  pre_crisis_sd   <- sd(pre_crisis_data$Beta_Mean, na.rm = TRUE)
  
  # 2. Crisis Period
  crisis_data <- df_beta_all_models %>%
    filter(Model == MODEL_TO_ANALYZE, 
           Tau_Level == tau_val,
           Date >= CRISIS_START, Date <= CRISIS_END)
  
  if (nrow(crisis_data) > 0) {
    if (tau_val < 0.5) { # Downside beta, look for trough
      crisis_extreme_val  <- min(crisis_data$Beta_Mean, na.rm = TRUE)
      crisis_extreme_date <- crisis_data$Date[which.min(crisis_data$Beta_Mean)]
    } else { # Upside beta, look for peak
      crisis_extreme_val  <- max(crisis_data$Beta_Mean, na.rm = TRUE)
      crisis_extreme_date <- crisis_data$Date[which.max(crisis_data$Beta_Mean)]
    }
  } else {
    crisis_extreme_val <- NA
    crisis_extreme_date <- NA
  }
  
  # 3. Post-Crisis Period
  post_crisis_data <- df_beta_all_models %>%
    filter(Model == MODEL_TO_ANALYZE, 
           Tau_Level == tau_val,
           Date >= POST_CRISIS_START, Date <= POST_CRISIS_END)
  
  post_crisis_mean <- mean(post_crisis_data$Beta_Mean, na.rm = TRUE)
  post_crisis_sd   <- sd(post_crisis_data$Beta_Mean, na.rm = TRUE)
  
  # 4. Magnitudes of Change
  magnitude_change_crisis <- crisis_extreme_val - pre_crisis_mean
  magnitude_change_perm   <- post_crisis_mean - pre_crisis_mean
  
  return(
    list(
      pre_crisis_mean = pre_crisis_mean,
      pre_crisis_sd = pre_crisis_sd,
      crisis_extreme_val = crisis_extreme_val,
      crisis_extreme_date = crisis_extreme_date,
      post_crisis_mean = post_crisis_mean,
      post_crisis_sd = post_crisis_sd,
      magnitude_change_crisis = magnitude_change_crisis,
      magnitude_change_perm = magnitude_change_perm
    )
  )
}

# --- Extract stats for both Downside and Upside Betas ---
downside_stats <- get_detailed_stats(0.05)
upside_stats   <- get_detailed_stats(0.95)

# --- Print the results clearly for copy-pasting ---
cat("--- Please copy ALL the following output and paste it back to me ---\n\n")

cat("--- DOWNSIDE BETA (tau = 0.05) ANALYSIS ---\n")
cat(sprintf("Pre-Crisis Mean (2019): %.4f\n", downside_stats$pre_crisis_mean))
cat(sprintf("Pre-Crisis Std. Dev. (2019): %.4f\n", downside_stats$pre_crisis_sd))
cat(sprintf("Crisis Trough Value (2020-21): %.4f\n", downside_stats$crisis_extreme_val))
cat(sprintf("Date of Crisis Trough: %s\n", as.character(downside_stats$crisis_extreme_date)))
cat(sprintf("Post-Crisis Mean (2022-onward): %.4f\n", downside_stats$post_crisis_mean))
cat(sprintf("Post-Crisis Std. Dev. (2022-onward): %.4f\n", downside_stats$post_crisis_sd))
cat(sprintf("Magnitude of Crisis Shock (Trough - Pre-Crisis Mean): %.4f\n", downside_stats$magnitude_change_crisis))
cat(sprintf("Magnitude of Persistent Shift (Post-Crisis Mean - Pre-Crisis Mean): %.4f\n\n", downside_stats$magnitude_change_perm))

cat("--- UPSIDE BETA (tau = 0.95) ANALYSIS ---\n")
cat(sprintf("Pre-Crisis Mean (2019): %.4f\n", upside_stats$pre_crisis_mean))
cat(sprintf("Pre-Crisis Std. Dev. (2019): %.4f\n", upside_stats$pre_crisis_sd))
cat(sprintf("Crisis Peak Value (2020-21): %.4f\n", upside_stats$crisis_extreme_val))
cat(sprintf("Date of Crisis Peak: %s\n", as.character(upside_stats$crisis_extreme_date)))
cat(sprintf("Post-Crisis Mean (2022-onward): %.4f\n", upside_stats$post_crisis_mean))
cat(sprintf("Post-Crisis Std. Dev. (2022-onward): %.4f\n", upside_stats$post_crisis_sd))
cat(sprintf("Magnitude of Crisis Shock (Peak - Pre-Crisis Mean): %.4f\n", upside_stats$magnitude_change_crisis))
cat(sprintf("Magnitude of Persistent Shift (Post-Crisis Mean - Pre-Crisis Mean): %.4f\n", upside_stats$magnitude_change_perm))

cat("\n\n--- All scripts executed successfully ---\n")
