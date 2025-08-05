This repository contains the source code for the numerical experiments in our paper "Bayesian Smoothed Quantile Regression". For a better reproduction experience, we provide a detailed illustration as follows:

## Repository Structure

```text
BSQR/
│
├── Simulation/
│   ├── bsqr_epanechnikov_Z_robust.stan
│   ├── bsqr_gaussian_Z_robust.stan
│   ├── bsqr_triangular_Z_robust.stan
│   ├── bsqr_uniform_Z_robust.stan
│   ├── simulation_main.R
│   └── BSQR_SimResults_AllKernels_20250715_M200_n200_FULLMCMC_CV.csv
│
├── Empirical analysis/
│   ├── bsqr_triangular_Z_robust.stan
│   ├── bsqr_uniform_Z_robust.stan
│   ├── bqr_ald.stan
│   ├── Empirical analysis.R
│   ├── SPX.csv
│   └── JPM.csv
│
├── LICENSE
│
└── README.md
```

## 1. Installation

To get started, you can either download the source code directly or use Git to clone the repository.

```bash
git clone https://github.com/BeauquinLau/BSQR.git
cd BSQR
```

Ensure you have R and RStudio installed. 

**Important:** `rstan` requires a working C++ compiler. Please follow the instructions at [RStan Getting Started](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started) to configure your system correctly.

## 2. Reproducing Numerical Results

This repository provides executable scripts for the simulation studies and the empirical analysis presented in our paper.

**General Instruction:** For each part (Simulation and Empirical Analysis), please ensure all files within the respective folder are downloaded and placed in the same directory. Set this directory as your working directory in R or RStudio before running the main script.

### 2.1. Simulation Study

This study provides a comprehensive evaluation of the BSQR framework’s performance. It assesses the estimation accuracy and inferential validity of BSQR under various settings and compares it against established frequentist and Bayesian methods. Synthetic data is generated from a linear model with covariates following an autoregressive covariance structure, considering both sparse, high-dimensional and dense, lower-dimensional coefficient scenarios. To evaluate robustness, four different error distributions are employed: standard normal, heavy-tailed Student’s t, a bimodal mixture normal, and heteroscedastic normal.

📌 **To run:** Navigate to the `Simulation/` directory and execute `simulation_main.R`.

```R
# In R/RStudio, with the working directory set to 'Simulation/'
source("simulation_main.R")
```

**`simulation_main.R`:**
*   Compiles the four `.stan` models for the different kernels.
*   Runs the Monte Carlo simulations to estimate model parameters under various settings.
*   Generates plots and summary tables for the simulation results.
*   The results from our run are provided in `BSQR_SimResults_AllKernels_20250715_M200_n200_FULLMCMC_CV.csv` for reference.

### 2.2. Empirical Analysis

This analysis demonstrates the empirical utility of the BSQR framework by investigating the asymmetric nature of dynamic systemic risk for a globally systemically important financial institution (JPMorgan Chase & Co.) in the post-COVID era. It applies a dynamic Capital Asset Pricing Model (CAPM) to the daily stock returns of JPM against the S&P 500 index. Using a rolling-window approach, the code estimates the downside beta (at τ=0.05) and upside beta (at τ=0.95) to capture how JPM’s risk exposure changes in different market conditions. The stability and economic insights from the BSQR models are compared against a standard Bayesian Quantile Regression (BQR-ALD) benchmark.

📌 **To run:** Navigate to the `Empirical analysis/` directory and execute `Empirical analysis.R`.

```R
# In R/RStudio, with the working directory set to 'Empirical analysis/'
source("Empirical analysis.R")
```
**`Empirical analysis.R`:**
*   Loads the financial time series data from `SPX.csv` and `JPM.csv`.
*   Fits the BSQR model (using Triangular and Uniform kernels) and the standard BQR model to the data.
*   Computes VaR estimates and generates comparison plots.
*   Produces the figures and tables presented in the empirical section of the paper.


## 3. Core Stan Models

*   **`bsqr_[kernel]_Z_robust.stan`**: Implements the BSQR model using different kernel functions (Gaussian, Uniform, Epanechnikov, Triangular) with a robust error structure.
*   **`bqr_ald.stan`**: Implements the standard Bayesian Quantile Regression model using the Asymmetric Laplace Distribution (ALD).


## 4. Dependencies

To reproduce the numerical results, please ensure you have a working R environment (version 4.0 or higher is recommended) and the following R packages are installed. The scripts are designed to automatically install any missing packages.

**Core Packages (Used in both scripts):**
*   **`dplyr`**: For data manipulation and transformation.
*   **`readr`**: For reading CSV files efficiently (used in Empirical analysis).
*   **`cmdstanr`**: The primary interface for fitting Stan models.
*   **`posterior`**: For processing and summarizing MCMC output.
*   **`quantreg`**: For running standard quantile regression as a benchmark and for initialization.
*   **`ggplot2`**: For generating plots and figures.
*   **`knitr`**: For generating LaTeX tables from R data frames.

**Additional Packages for Empirical Analysis:**
*   **`lubridate`**: For handling date objects.
*   **`tidyr`**: For data tidying, used in plotting.
*   **`caret`**: For creating cross-validation data folds.
*   **`e1071`**: For calculating skewness and kurtosis.
*   **`stringr`**: For string manipulation in file and plot naming.
*   **`colorspace`**: For creating advanced color palettes for plots.

**Additional Packages for Simulation:**
*   **`MASS`**: For generating multivariate normal data.
*   **`rstan`**: While `cmdstanr` is used for fitting, some functions might rely on `rstan`'s ecosystem.
*   **`matrixStats`**: For efficient matrix calculations.
*   **`brms`**: For fitting the Bayesian Quantile Regression (BQR-ALD) benchmark model.
*   **`parallel`**, **`future`**, **`future.apply`**, **`progressr`**: For parallel execution of the simulation replications.

## 5. Citation

If you use this code or the BSQR methodology in your research, please consider citing our paper:

```bibtex
@article{Liu2025,
  author    = {Liu, Bingqi and Li, Kangqiang and Pang, Tianxiao},
  title     = {Bayesian Smoothed Quantile Regression},
  journal   = {arXiv preprint arXiv:2508.01738},
  year      = {2025},
  doi       = {10.48550/arXiv.2508.01738},
  url       = {http://arxiv.org/abs/2508.01738},
}
```
