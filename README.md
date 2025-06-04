# EfficientEstimationDSGE

This repository contains MATLAB code accompanying the paper  
**"Efficient Estimation of Nonlinear DSGE Models"**  
by *Eva Janssens* and *Sean McCrary*.

Each subdirectory implements a specific model with its own estimation or filtering routine. The general structure of the filtering procedure alternates between:

1. **Solution Step**: Solves the model locally at the current state forecast.
2. **Filtering Step**: Constructs state-space matrices consistent with the current local solution and performs a single step of the time-varying Kalman filter.

Each subdirectory contains a dedicated `README.md` with details on the model, data, and estimation or filtering demonstration.


## 📁 Contents

### [NKDMP](NKDMP)
MATLAB code for full-information Bayesian estimation of the **New Keynesian Diamond-Mortensen-Pissarides (NKDMP)** model using a Random-Walk Metropolis-Hastings sampler.

- Three policy variables: market tightness, inflation, marginal cost  
- Five structural shocks: productivity, discount factor, monetary policy, matching efficiency, separation rate  
- Estimation on U.S. data from 1966:Q1 to 2019:Q4  
- Newton solver with symbolic equilibrium conditions

---

### [DMP](DMP)
MATLAB code for estimating the **Diamond-Mortensen-Pissarides (DMP)** model.

- Uses data simulated from the global solution as the data-generating process  
- Benchmarks the computational performance of existing filters
  
---

### [TextbookNewKeynesian](TextbookNewKeynesian)
MATLAB code for estimating the **standard three-equation New Keynesian model**.

- Solves the textbook three-equation New Keynesian model  
- Full-information Bayesian estimation using data from 1966:Q1 to 2007:Q4 

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
