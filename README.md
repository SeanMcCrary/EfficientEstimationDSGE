# EfficientEstimationDSGE

This repository contains MATLAB code for efficient estimation and filtering of various DSGE models. The models span labor market search, New Keynesian dynamics, and stochastic growth. Each subdirectory implements a specific model with its own estimation or filtering routine.

## 📁 Contents

### [NKDMP](NKDMP)
MATLAB code for full-information Bayesian estimation of the **New Keynesian Diamond-Mortensen-Pissarides (NKDMP)** model using a Random-Walk Metropolis-Hastings sampler.

- Three policy variables: market tightness, inflation, marginal cost  
- Five structural shocks: productivity, discount factor, monetary policy, matching efficiency, separation rate  
- Estimation on U.S. data from 1966:Q1 to 2019:Q4  
- Newton solver with symbolic equilibrium conditions

---

### [DMP](DMP)
MATLAB code for estimating the **Diamond-Mortensen-Pissarises (DMP)** model.

- Data simulated from the global solution is used as the data-generating process  
- Benchmarks computational times of existing filters  
- Estimation uses a Random-Walk Metropolis-Hastings algorithm

---

### [TextbookNewKeynesian](TextbookNewKeynesian)
MATLAB code for estimating the **standard three-equation New Keynesian model**.

- Implements textbook specification  
- Full-information Bayesian estimation

---

### [StochasticGrowth_AR1SV](StochasticGrowth_AR1SV)
MATLAB code for a **stochastic growth model with AR(1) productivity** and **stochastic volatility**.

- Demonstrates how to implement the filtering procedure when the volatility of productivity follows an AR(1) process

---

### [StochasticGrowth_RegimeSwitching](StochasticGrowth_RegimeSwitching)
MATLAB code for a **stochastic growth model with regime-switching volatility**.

- Demonstrates the filtering procedure when productivity dynamics are determined by a discrete Markov process

---

## 🛠 Requirements

- MATLAB R2024a or later  
- Symbolic Math Toolbox  
- Statistics and Machine Learning Toolbox

---

## 🔍 Contact

This is a work in progress. Code will be updated periodically.  
For questions, suggestions, or concerns, feel free to reach out: mccrary.65 **at** osu.edu



---
