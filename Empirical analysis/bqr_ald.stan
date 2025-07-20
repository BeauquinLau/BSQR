functions {
  real asym_laplace_lpdf(real y, real mu, real sigma, real tau) {
    real err = y - mu;
    real sign = (err > 0) - (err < 0);
    return log(tau * (1 - tau)) - log(sigma) - rho_tau(err, tau) / sigma;
  }
  
  real rho_tau(real err, real tau) {
      if (err >= 0) {
        return tau * err;
      } else {
        return (tau - 1) * err;
      }
  }
}

data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] X;
  vector[N] y;
  real<lower=0, upper=1> tau;

  // Prior parameters
  real beta_loc;
  real<lower=0> beta_scale;
  real<lower=0> sigma_cauchy_scale;
}

parameters {
  vector[K] beta;
  real<lower=0> sigma;
}

model {
  // Priors
  beta ~ normal(beta_loc, beta_scale); 
  sigma ~ cauchy(0, sigma_cauchy_scale);

  // Likelihood
  for (n in 1:N) {
    y[n] ~ asym_laplace(X[n] * beta, sigma, tau);
  }
}