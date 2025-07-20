functions {
  // Smoothed Loss Function L_h(u; tau, h) for Epanechnikov Kernel
  real smoothed_loss_epanechnikov(real u, real tau, real h) {
    if (u <= -h) {
      return (tau - 1.0) * u;
    } else if (u >= h) {
      return tau * u;
    } else {
      // Formula for -h < u < h
      real e = u;
      return (3.0 * e^2) / (8.0 * h) - (e^4) / (16.0 * h^3) + e * (tau - 0.5) + (3.0 * h) / 16.0;
    }
  }
  // Integrand for the normalizing constant Z for the Epanechnikov kernel
  real integrand_for_Z_epanechnikov(real u, real not_used_xc,
                                    array[] real theta_real_array,
                                    data array[] real x_r,
                                    data array[] int x_i) {
    real theta_val = theta_real_array[1];
    real tau_val = theta_real_array[2];
    real h_val = theta_real_array[3];
    return exp(-theta_val * smoothed_loss_epanechnikov(u, tau_val, h_val));
  }
  // ROBUST log(Z) calculator for Epanechnikov Kernel
  real calculate_log_Z_epanechnikov(real theta_val,
                                    data real tau_val,
                                    data real h_val,
                                    data real Z_rel_tol_val,
                                    data real K_asymptotic_switch,
                                    data int use_asymptotic_approx,
                                    data array[] real x_r_data,
                                    data array[] int x_i_data) {
    if (abs(tau_val - 0.5) < 1e-9) {
      if (use_asymptotic_approx == 1 && theta_val > K_asymptotic_switch) {
        return log(4) - log(theta_val);
      } else {
        real integral_val_center = integrate_1d(integrand_for_Z_epanechnikov,
                                                -h_val, h_val,
                                                {theta_val, tau_val, h_val}, x_r_data, x_i_data, Z_rel_tol_val);
        real log_I_tail = theta_val * h_val * 0.5 - log(theta_val * 0.5);
        if (integral_val_center > 1e-12) {
          return log_sum_exp([log_I_tail, log(integral_val_center), log_I_tail]);
        } else {
          return log_sum_exp([log_I_tail, log_I_tail]);
        }
      }
    }
    else {
      if (use_asymptotic_approx == 1 && (theta_val * h_val > K_asymptotic_switch)) {
        real term = square(tau_val - 0.5) + 1.0;
        return -0.5 * log(theta_val) + 0.5 * log(2.0 * pi() / term);
      } else {
        real integral_val_center = integrate_1d(integrand_for_Z_epanechnikov,
                                                -h_val, h_val,
                                                {theta_val, tau_val, h_val}, x_r_data, x_i_data, Z_rel_tol_val);
        real log_I1_left_tail = theta_val * h_val * (1.0 - tau_val) - log(theta_val * (1.0 - tau_val));
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
  int<lower=0> N_train_obs;
  int<lower=1> K;
  matrix[N_train_obs, K] X_train;
  vector[N_train_obs] y_train;
  
  real<lower=0, upper=1> tau;
  real<lower=0> h;

  vector[K] beta_location;
  vector<lower=0>[K] beta_scale;
  
  real<lower=0> gamma_shape;
  real<lower=0> gamma_rate;
  real<lower=0> upper_bound_for_theta;

  // --- These are specific to the Epanechnikov Z calculation logic ---
  real Z_outer_integration_lower_bound;
  real Z_outer_integration_upper_bound;
  
  real<lower=0> epsilon_theta;
  real<lower=0> Z_rel_tol;
  
  real<lower=0> K_ASYMPTOTIC_SWITCH_STD_DEVS;
  int<lower=0, upper=1> USE_Z_ASYMPTOTIC_APPROX;
}

transformed data {
  // --- Required for the integrator, part of original code structure ---
  array[0] real x_r_empty;
  array[0] int x_i_empty;
}

parameters {
  // --- Parameters are set up for NCP ---
  vector[K] beta_raw;
  real<lower=epsilon_theta, upper=upper_bound_for_theta> theta;
}

transformed parameters {
  // --- This block implements the Non-Centered Parameterization ---
  vector[K] beta = beta_location + beta_raw .* beta_scale;
}

model {
  beta_raw ~ std_normal();
  theta ~ gamma(gamma_shape, gamma_rate);

  real log_Z_val = calculate_log_Z_epanechnikov(theta, tau, h,
                                                Z_rel_tol,
                                                K_ASYMPTOTIC_SWITCH_STD_DEVS,
                                                USE_Z_ASYMPTOTIC_APPROX,
                                                x_r_empty,
                                                x_i_empty);

  vector[N_train_obs] u_residuals = y_train - X_train * beta;
  real sum_neg_theta_Lh = 0;

  for (i in 1:N_train_obs) {
    sum_neg_theta_Lh += -theta * smoothed_loss_epanechnikov(u_residuals[i], tau, h);
  }

  if (log_Z_val < -1e9 || is_inf(log_Z_val) || is_nan(log_Z_val)) {
    target += negative_infinity(); // Reject the step cleanly if Z calculation failed
  } else {
    if (N_train_obs > 0) {
      target += sum_neg_theta_Lh - N_train_obs * log_Z_val;
    }
  }
}
// End of Stan Code
