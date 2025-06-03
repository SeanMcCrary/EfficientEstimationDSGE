# Stochastic Growth Model with Regime Switching

This repository contains MATLAB code for a **stochastic growth model** with **regime switching** in productivity dynamics. The model is solved using a local projection method and a time-varying Kalman filter with regime-dependent policy functions.

## Overview

- The model features a 2-state Markov chain governing exogenous regimes.
- Each regime has its own AR(1) process for productivity with different persistence and volatility.
- Policy functions are locally approximated using a Newton solver.
- A switching Kalman filter is used to infer latent states and regime probabilities from simulated data.

## File Descriptions

- `demo.m`  
  Demonstrates the full simulation and filtering procedure on synthetic data.

- `rbcar1mc_tpcoef.m`  
  Computes regime-specific linear policy coefficients at a given state using Newton's method. Called during simulation and filtering.

- `Res_eval.m` and `Jac_eval.m`  
  Contain the residual and Jacobian functions used in the Newton solver. These are automatically generated.

- `rbcar1mc_symbolic_equations.m`  
  Constructs the symbolic model equations and generates `Res_eval.m` and `Jac_eval.m` using MATLAB's symbolic toolbox.

## Requirements

- MATLAB (we use 2024a) 
- Symbolic Math Toolbox
- Statistics and Machine Learning Toolbox (for `mvnpdf` and `dtmc`)

