# Stochastic Growth Model with Stochastic Volatility

This repository contains MATLAB code for a **stochastic growth model** with **stochastic volatility** in productivity. The model is solved using Taylor projection and two filtering approaches are presented:

1. Extended Kalman Filter (EKF)  
2. Discretized filter for latent volatility

## Overview

- Productivity follows an AR(1) process with stochastic volatility (log-AR(1)).
- Policy functions are solved via a Newton method.
- Filtering recovers latent states and regime probabilities from simulated data.

## Files

- `demo_EKF.m`  
  Runs simulation and EKF-based filtering on synthetic data.

- `demo_discretization_filter.m`  
  Runs simulation and filtering using a discretized latent volatility process.

- `rbcar1sv_tpcoef.m`  
  Computes linear policy coefficients at a given state via Newton's method.

- `Res_eval.m`, `Jac_eval.m`  
  Residual and Jacobian functions used by the Newton solver. Auto-generated by `rbcar1sv_symbolic_equations.m`.

- `rbcar1sv_symbolic_equations.m`  
  Builds symbolic model equations and generates `Res_eval.m` and `Jac_eval.m`.

- `normal_gh_quadrature.m`  
  Constructs Gauss-Hermite nodes and weights for standard normal integration. Used in equilibrium condition specification.

- `mtauchen.m`  
  Approximates an AR(1) process with a finite-state Markov chain. Used by `demo_discretization_filter.m`.

## Requirements

- MATLAB (tested with R2024a)  
- Symbolic Math Toolbox  
- Statistics and Machine Learning Toolbox
