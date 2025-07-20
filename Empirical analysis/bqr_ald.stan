// 文件名: bqr_ald.stan
// 描述: 传统的贝叶斯分位数回归模型，使用非对称拉普拉斯分布(ALD)
// 版本：v2 (【最终修正版】已添加asym_laplace_lpdf的函数定义)

// 【【【这是关键的、唯一的修改】】】
// Stan本身没有内置asym_laplace分布, 我们需要手动定义它的对数概率密度函数(lpdf)
// 这个定义是标准的，来源于brms的实现和Stan官方文档的示例
functions {
  real asym_laplace_lpdf(real y, real mu, real sigma, real tau) {
    real err = y - mu;
    real sign = (err > 0) - (err < 0);
    return log(tau * (1 - tau)) - log(sigma) - rho_tau(err, tau) / sigma;
  }
  
  // asym_laplace_lpdf 依赖于 check/pinball loss 函数 rho_tau
  // 我们也需要定义它
  real rho_tau(real err, real tau) {
      if (err >= 0) {
        return tau * err;
      } else {
        return (tau - 1) * err;
      }
  }
}

data {
  int<lower=0> N;                // 观测数量
  int<lower=0> K;                // 回归系数数量 (包括截距项)
  matrix[N, K] X;              // 设计矩阵 (自变量)
  vector[N] y;                 // 因变量
  real<lower=0, upper=1> tau;  // 目标分位数

  // 先验参数
  real beta_loc;                 // beta先验的均值 (通常为0)
  real<lower=0> beta_scale;      // beta先验的标准差
  real<lower=0> sigma_cauchy_scale; // sigma先验的尺度
}

parameters {
  vector[K] beta;              // 回归系数 (alpha 和 beta)
  real<lower=0> sigma;         // 尺度参数
}

model {
  // 先验 (Priors)
  beta ~ normal(beta_loc, beta_scale); 
  sigma ~ cauchy(0, sigma_cauchy_scale);

  // 似然 (Likelihood)
  // 我们仍然需要for循环，因为我们定义的asym_laplace_lpdf也是处理单个观测的
  for (n in 1:N) {
    y[n] ~ asym_laplace(X[n] * beta, sigma, tau);
  }
}
