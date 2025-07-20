functions {
  // Smoothed Loss Function L_h(u; tau, h) for Uniform Kernel
  real smoothed_loss_uniform(real u, real h, real tau) {
    if (u < -h) {
      return (tau - 1.0) * u;
    } else if (u > h) {
      return tau * u;
    } else {
      // Parabolic smoothing in the interval [-h, h]
      return u^2 / (4.0 * h) + (tau - 0.5) * u + h / 4.0;
    }
  }
}

data {
  // --- Data block names are UNIFIED with the R script and other kernels ---
  int<lower=0> N_train_obs; // Allowed to be 0 for prior predictive checks
  int<lower=1> K;
  
  matrix[N_train_obs, K] X_train;
  vector[N_train_obs] y_train;
  
  real<lower=0, upper=1> tau;
  real<lower=0> h;

  vector[K] beta_location;
  vector<lower=0>[K] beta_scale;
  real<lower=0> gamma_shape;
  real<lower=0> theta_prior_rate_val;
  real<lower=0> upper_bound_for_theta;
  real<lower=0> epsilon_theta;
}

parameters {
  vector[K] beta_raw; 
  real<lower=epsilon_theta, upper=upper_bound_for_theta> theta;
}

transformed parameters {
  vector[K] beta = beta_location + beta_raw .* beta_scale;
  
  real log_Z_val;
  {
    real local_h = h;
    real local_tau = tau;
    real local_theta = theta;

    if (local_h < 1e-9) {
      log_Z_val = -log(local_theta) - log(local_tau) - log(1.0 - local_tau);
    } else {
      // Term 1: Integral over u > h
      real log_I3 = -local_theta * local_tau * local_h - log(local_theta * local_tau);

      // Term 2: Integral over u < -h
      real log_I1 = local_theta * (local_tau - 1.0) * local_h - log(theta * (1.0 - local_tau));

      // Term 3: Integral over -h <= u <= h
      real a = local_theta / (4.0 * local_h);
      real b = local_theta * (local_tau - 0.5);
      real c = local_theta * local_h / 4.0;
      
      real mu = -b / (2.0 * a); 
      real s = sqrt(1.0 / (2.0 * a));
      
      real log_scaling_factor = 0.5 * log(2.0 * pi()) + log(s) + (b^2 / (4.0 * a)) - c;
      real log_prob_mass = log_diff_exp(std_normal_lcdf((local_h - mu) / s), std_normal_lcdf((-local_h - mu) / s));
      real log_I2 = log_scaling_factor + log_prob_mass;
      
      log_Z_val = log_sum_exp([log_I1, log_I2, log_I3]);
    }
  }
}

model {
  // --- Priors use unified parameter names ---
  beta_raw ~ std_normal(); 
  theta ~ gamma(gamma_shape, theta_prior_rate_val);

  // --- Likelihood uses unified variable names ---
  // The 'if' statement to protect the calculation is correctly placed here.
  if (is_inf(log_Z_val) || is_nan(log_Z_val) || log_Z_val > 1e9) {
    target += negative_infinity(); // Reject step if Z calculation failed
  } else {
    if (N_train_obs > 0) {
      vector[N_train_obs] residuals = y_train - X_train * beta;
      real sum_neg_theta_Lh = 0;
      
      for(i in 1:N_train_obs) {
        sum_neg_theta_Lh += -theta * smoothed_loss_uniform(residuals[i], h, tau);
      }
      target += sum_neg_theta_Lh - N_train_obs * log_Z_val;
    }
  }
}
// End of Stan Code
