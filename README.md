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
├── Empirical_Analysis/
│   ├── bsqr_triangular_Z_robust.stan
│   ├── bsqr_uniform_Z_robust.stan
│   ├── bqr_ald.stan
│   ├── Empirical_analysis.R
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

This experiment evaluates the performance of the Bayesian Smoothed Quantile Regression (BSQR) model with different kernel functions (Epanechnikov, Gaussian, Triangular, Uniform) through Monte Carlo simulations.

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

This analysis applies the BSQR model to real-world financial data (S&P 500 and JPMorgan Chase & Co.) to estimate Value-at-Risk (VaR), demonstrating the model's practical utility. The analysis compares BSQR with a standard Bayesian Quantile Regression (BQR) model.

📌 **To run:** Navigate to the `Empirical_Analysis/` directory and execute `Empirical_analysis.R`.

```R
# In R/RStudio, with the working directory set to 'Empirical_Analysis/'
source("Empirical_analysis.R")
```
**`Empirical_analysis.R`:**
*   Loads the financial time series data from `SPX.csv` and `JPM.csv`.
*   Fits the BSQR model (using Triangular and Uniform kernels) and the standard BQR model to the data.
*   Computes VaR estimates and generates comparison plots.
*   Produces the figures and tables presented in the empirical section of the paper.


## 3. Core Stan Models

*   **`bsqr_[kernel]_Z_robust.stan`**: Implements the BSQR model using different kernel functions (Epanechnikov, Gaussian, Triangular, Uniform) with a robust error structure.
*   **`bqr_ald.stan`**: Implements the standard Bayesian Quantile Regression model using the Asymmetric Laplace Distribution (ALD).


## 4. Dependencies

To reproduce the numerical results, please ensure the following R packages are installed. You can check your versions in R.

*   **rstan**: `(e.g., version 2.21.0 or higher)`
*   **ggplot2**: `(e.g., version 3.4.0 or higher)`
*   **dplyr**: `(e.g., version 1.1.0 or higher)`
*   **reshape2**: `(e.g., version 1.4.4 or higher)`
*   **gridExtra**: `(e.g., version 2.3 or higher)`

*(Note: Please replace the example versions with the actual versions you used if known, or leave as is.)*
