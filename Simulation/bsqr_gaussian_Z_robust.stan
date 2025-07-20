functions {
  // Stan's own normal cdf can be slow; this is a standard, fast approximation.
  real Phi_approx_custom(real x) {
    return inv_logit(0.07056 * x^3 + 1.5976 * x);
  }

  // Smoothed Loss Function L_h(u; tau, h) for Gaussian Kernel
  real smoothed_loss_gaussian(real u, real tau, real h) {
    real h_eff = fmax(h, 1e-9);
    real u_over_h = u / h_eff;
    // Calling the renamed custom function.
    real cdf_val = Phi_approx_custom(u_over_h);
    real pdf_val = exp(std_normal_lpdf(u_over_h));
    real Lh = u * (cdf_val - (1.0 - tau)) + h_eff * pdf_val;
    return fmax(0.0, Lh); // Ensure non-negativity
  }

  // Integrand for the normalizing constant Z
  real integrand_for_Z_gaussian(real u, real not_used_xc,
                                array[] real theta_real_array,
                                data array[] real x_r,
                                data array[] int x_i) {
    real theta_val = theta_real_array[1];
    real tau_val = theta_real_array[2];
    real h_val = theta_real_array[3];
    return exp(-theta_val * smoothed_loss_gaussian(u, tau_val, h_val));
  }

  // Robust log(Z) calculator
  real calculate_log_Z_gaussian(real theta_val,
                                data real tau_val,
                                data real h_val,
                                data real Z_rel_tol_val,
                                data real K_asymptotic_switch,
                                data int use_asymptotic_approx,
                                data array[] real x_r_data,
                                data array[] int x_i_data) {

    // --- STRATEGY 1: Handle the special case of tau = 0.5 ---
    if (abs(tau_val - 0.5) < 1e-9) {
      if (use_asymptotic_approx == 1 && theta_val > K_asymptotic_switch) {
        return log(4) - log(theta_val);
      } else {
        real half_width = 15.0 * h_val + 5.0;
        array[3] real params = {theta_val, tau_val, h_val};
        real integral_val = integrate_1d(integrand_for_Z_gaussian,
                                         -half_width, half_width,
                                         params, x_r_data, x_i_data, Z_rel_tol_val);
        if (integral_val > 1e-12) {
          return log(integral_val);
        } else {
          return -1e10; // Integration failed
        }
      }
    }
    // --- STRATEGY 2 & 3: Handle all other tau values ---
    else {
      if (use_asymptotic_approx == 1 && (theta_val * h_val > K_asymptotic_switch)) {
        real term = square(tau_val - 0.5) + 1.0;
        return -0.5 * log(theta_val) + 0.5 * log(2.0 * pi() / term);
      } else {
        real center = h_val * inv_Phi(1.0 - tau_val);
        real width = 15.0 * h_val + 5.0;

        array[3] real params = {theta_val, tau_val, h_val};
        real integral_val = integrate_1d(integrand_for_Z_gaussian,
                                         center - width, center + width,
                                         params, x_r_data, x_i_data, Z_rel_tol_val);

        if (integral_val > 1e-12) {
          return log(integral_val);
        } else {
          return -1e10; // Integration failed
        }
      }
    }
  }
}

data {
  int<lower=0> N_train_obs;
  int<lower=1> K;
  matrix[N_train_obs, K] X_train;
  vector[N_train_obs] y_train;

  real<lower=0, upper=1> tau;
  real<lower=0> h;

  // Priors for NCP and theta
  vector[K] beta_location;
  vector<lower=0>[K] beta_scale;
  real<lower=0> gamma_shape;
  real<lower=0> theta_prior_rate_val;

  real<lower=0> upper_bound_for_theta;
  real<lower=0> epsilon_theta;
  real<lower=0> Z_rel_tol;
  real<lower=0> K_ASYMPTOTIC_SWITCH_STD_DEVS;
  int<lower=0, upper=1> USE_Z_ASYMPTOTIC_APPROX;
}

transformed data {
  array[0] real x_r_empty;
  array[0] int x_i_empty;
}

parameters {
  // beta_params is now beta_raw for Non-Centered Parameterization
  vector[K] beta_raw;
  // theta_param is now theta
  real<lower=epsilon_theta, upper=upper_bound_for_theta> theta;
}

transformed parameters {
  // This block implements NCP and pre-calculates for efficiency
  
  // 1. Transform beta_raw back to the centered beta_params
  vector[K] beta_params = beta_location + beta_scale .* beta_raw;

  // 2. Pre-calculate the log(Z) normalizing constant
  real log_Z_val = calculate_log_Z_gaussian(theta, tau, h,
                                            Z_rel_tol,
                                            K_ASYMPTOTIC_SWITCH_STD_DEVS,
                                            USE_Z_ASYMPTOTIC_APPROX,
                                            x_r_empty,
                                            x_i_empty);

  // 3. Pre-calculate the sum of the smoothed loss term
  vector[N_train_obs] u_residuals = y_train - X_train * beta_params;
  real sum_neg_theta_Lh = 0;
  if (N_train_obs > 0) {
    for (i in 1:N_train_obs) {
      sum_neg_theta_Lh += -theta * smoothed_loss_gaussian(u_residuals[i], tau, h);
    }
  }
}

model {
  // Priors are now on the non-centered/unified parameters
  beta_raw ~ std_normal();
  theta ~ gamma(gamma_shape, theta_prior_rate_val);

  // The log-likelihood is now calculated using the pre-computed values
  // from the transformed parameters block.
  if (log_Z_val < -1e9) {
    target += negative_infinity(); // Reject step if Z calculation failed
  } else {
    if (N_train_obs > 0) {
      target += sum_neg_theta_Lh - N_train_obs * log_Z_val;
    }
  }
}
// End of Stan Code
