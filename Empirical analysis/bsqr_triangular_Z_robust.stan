functions {
  // Smoothed Loss Function L_h(e; tau, h) for Triangular Kernel
  real smoothed_loss_triangular(real e, real tau, real h) {
    real u = e / h;
    if (u <= -1.0) {
      return (tau - 1.0) * e;
    } else if (u < 0.0) {
      return h / 6.0 * pow(1.0 + u, 3) - e * (1.0 - tau);
    } else if (u < 1.0) {
      return e * tau + h / 6.0 * pow(1.0 - u, 3);
    } else { // u >= 1.0
      return tau * e;
    }
  }

  // Integrand for the normalizing constant Z for the Triangular kernel
  real integrand_for_Z_triangular(real u, real not_used_xc, 
                                    array[] real theta_real_array, 
                                    data array[] real x_r, 
                                    data array[] int x_i) {
    real theta_val = theta_real_array[1];
    real tau_val = theta_real_array[2];
    real h_val = theta_real_array[3];
    return exp(-theta_val * smoothed_loss_triangular(u, tau_val, h_val));
  }

  // ROBUST log(Z) calculator for Triangular Kernel
  real calculate_log_Z_triangular(real theta_val,
                                    data real tau_val, 
                                    data real h_val,
                                    data real Z_rel_tol_val, 
                                    data real K_asymptotic_switch,
                                    data int use_asymptotic_approx,
                                    data array[] real x_r_data,
                                    data array[] int x_i_data) {
    
    // Asymptotic approximation for tau=0.5 (Symmetric)
    if (abs(tau_val - 0.5) < 1e-9) {
      if (use_asymptotic_approx == 1 && theta_val > K_asymptotic_switch) {
        return log(4) - log(theta_val);
      } else {
        // Full integral for symmetric case
        real integral_val_center = integrate_1d(integrand_for_Z_triangular, 
                                                -h_val, h_val, 
                                                {theta_val, tau_val, h_val}, x_r_data, x_i_data, Z_rel_tol_val);
        real log_I_tail = -theta_val * h_val * 0.5 - log(theta_val * 0.5);
        if (integral_val_center > 1e-12) {
          return log_sum_exp([log_I_tail, log(integral_val_center), log_I_tail]);
        } else {
          return log_sum_exp([log_I_tail, log_I_tail]);
        }
      }
    } 
    // Asymptotic approximation for tau != 0.5 (Asymmetric)
    else {
      if (use_asymptotic_approx == 1 && (theta_val * h_val > K_asymptotic_switch)) {
        real term = square(tau_val - 0.5) + 1.0;
        return -0.5 * log(theta_val) + 0.5 * log(2.0 * pi() / term);
      } else {
        // Full integral for asymmetric case
        real integral_val_center = integrate_1d(integrand_for_Z_triangular, 
                                                -h_val, h_val,
                                                {theta_val, tau_val, h_val}, x_r_data, x_i_data, Z_rel_tol_val);
        real log_I1_left_tail = -theta_val * h_val * (1.0 - tau_val) - log(theta_val * (1.0 - tau_val));
        real log_I3_right_tail = -theta_val * h_val * tau_val - log(theta_val * tau_val);
        if (integral_val_center > 1e-12) {
          return log_sum_exp([log_I1_left_tail, log(integral_val_center), log_I3_right_tail]);
        } else {
          return log_sum_exp([log_I1_left_tail, log_I3_right_tail]);
        }
      }
    }
  }
}

data {
  // Core data
  int<lower=0> N_train_obs;
  int<lower=1> K; 
  matrix[N_train_obs, K] X_train; 
  vector[N_train_obs] y_train;
  
  // Model parameters
  real<lower=0, upper=1> tau;
  real<lower=0> h;

  // NCP prior parameters for beta
  vector[K] beta_location;
  vector<lower=0>[K] beta_scale;
  
  // Prior parameters for theta
  real<lower=0> gamma_shape; 
  real<lower=0> gamma_rate;  
  real<lower=0> upper_bound_for_theta;
  real<lower=0> epsilon_theta;

  // Robust Z calculation parameters
  real<lower=0> Z_rel_tol;
  real<lower=0> K_ASYMPTOTIC_SWITCH_STD_DEVS;
  int<lower=0, upper=1> USE_Z_ASYMPTOTIC_APPROX;
}

transformed data {
  // Empty arrays for integrate_1d, which requires them
  array[0] real x_r_empty;
  array[0] int x_i_empty;
}

parameters {
  vector[K] beta_raw; 
  real<lower=epsilon_theta, upper=upper_bound_for_theta> theta;
}

// Using the more efficient transformed parameters block structure
transformed parameters {
  vector[K] beta = beta_location + beta_scale .* beta_raw;
  vector[N_train_obs] residuals = y_train - X_train * beta;
  
  real sum_neg_theta_Lh = 0;
  for (i in 1:N_train_obs) {
    sum_neg_theta_Lh += -theta * smoothed_loss_triangular(residuals[i], tau, h);
  }

  real log_Z_val = calculate_log_Z_triangular(theta, tau, h, 
                                                Z_rel_tol,
                                                K_ASYMPTOTIC_SWITCH_STD_DEVS,
                                                USE_Z_ASYMPTOTIC_APPROX,
                                                x_r_empty, 
                                                x_i_empty);
}

model {
  // Priors
  beta_raw ~ std_normal(); 
  theta ~ gamma(gamma_shape, gamma_rate);

  // Likelihood
  if (is_inf(log_Z_val) || is_nan(log_Z_val) || log_Z_val < -1e9) {
    target += negative_infinity(); // Reject step cleanly if Z calculation failed
  } else {
    if (N_train_obs > 0) {
      target += sum_neg_theta_Lh - N_train_obs * log_Z_val;
    }
  }
}
// End of Stan Code
