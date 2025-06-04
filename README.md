# EfficientEstimationDSGE

This repository provides MATLAB implementations for efficient full-information estimation of **nonlinear DSGE models** using projection methods, symbolic computation, and Bayesian inference. Each subdirectory contains a self-contained model application, illustrating how to estimate complex macroeconomic models using modern solution and filtering techniques.

---

## 📁 Contents

### [`/NKDMP`](./tree/main/NKDMP)
Estimation of the **New Keynesian Diamond-Mortensen-Pissarides (NKDMP)** model with:
- Search and matching frictions,
- Nonlinear solution via Taylor projection,
- Filtering with the Time-Varying Kalman Filter (TV-KF),
- Bayesian estimation using Random-Walk Metropolis-Hastings.

> Includes symbolic residual and Jacobian systems with ZLB constraints.

---

### [`/DMP`](./tree/main/DMP)
Bayesian estimation of the baseline **Diamond-Mortensen-Pissarides (DMP)** labor market model.
- Nonlinear policy functions estimated from data on employment and market tightness.
- Uses Random-Walk Metropolis-Hastings and projection-based solution methods.

---

### [`/TextbookNewKeynesian`](./tree/main/TextbookNewKeynesian)
Estimation of the **canonical three-equation New Keynesian model**, as found in textbooks.
- Useful for benchmarking or comparison against nonlinear models.
- Implements full-information Bayesian estimation with a Kalman filter.

---

### [`/StochasticGrowth_AR1SV`](./tree/main/StochasticGrowth_AR1SV)
A **stochastic growth model** with:
- AR(1) productivity shocks,
- Time-varying stochastic volatility,
- Bayesian estimation using nonlinear filters.

---

### [`/StochasticGrowth_RegimeSwitching`](./tree/main/StochasticGrowth_RegimeSwitching)
A **stochastic growth model with regime-switching volatility**, modeling:
- Discrete changes in volatility regimes,
- Hidden Markov structure,
- Full-information filtering and likelihood evaluation.

---

## 🛠 Requirements

- MATLAB R2024a or later
- Symbolic Math Toolbox
- Statistics and Machine Learning Toolbox

---

## 📖 Citation

If you use this code in your research, please cite the relevant model-specific application. A full BibTeX entry will be provided upon publication.

---

## 🔍 Contact

For questions, suggestions, or collaborations, feel free to open an [issue](https://github.com/SeanMcCrary/EfficientEstimationDSGE/issues) or reach out via GitHub.

---
