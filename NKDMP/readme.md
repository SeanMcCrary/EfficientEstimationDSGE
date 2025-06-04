# NKDMP Model

This repository contains MATLAB code for full-information Bayesian estimation of the **New Keynesian Diamond-Mortensen-Pissarides (NKDMP)** model using a Random-Walk Metropolis-Hastings sampler.

## Overview

- The model includes three policy variables:
  - **Market tightness**
  - **Inflation**
  - **Marginal cost**
- And five structural shocks:
  - **Productivity shock**
  - **Discount factor shock**
  - **Monetary policy shock**
  - **Matching efficiency shock**
  - **Separation rate shock**
- Estimation is based on U.S. data for inflation, employment, market tightness, nominal interest rate, and job-finding rate, spanning 1966:Q1 to 2019:Q4.
- Policy functions are computed using a Newton solver.

## Files

- `demo_estimation.m`, `demo_data.mat`  
  Main script for running the estimation using the Random-Walk Metropolis-Hastings algorithm. Produces posterior draws and summary plots.

- `tpfilter_zlb.m`, `nk_zlb_tpcoef.m`  
  Functions used to filter the data and compute policy function coefficients via the Newton solver.

- `Rez_eval_zlb.m`, `Jac_eval_zlb.m`  
  Functions called by the Newton solver to evaluate residuals and Jacobians for the policy coefficients.

- `NomR_zlb.m`, `NomR_dx_zlb.m`  
  Functions called by the filter that output the linearized nominal interest rate.

- `nkdmp_zlb_symbolic_equations.m`  
  Symbolic script that constructs the model's equilibrium conditions and generates the functions:  
  `Rez_eval_zlb.m`, `Jac_eval_zlb.m`, `NomR_zlb.m`, and `NomR_dx_zlb.m`.

- `transformParamsInvME.m`, `transformParamsME.m`  
  Functions for mapping between parameter vectors and structural parameters, enforcing constraints such as positivity of variances.

## Requirements

- MATLAB (tested with R2024a)
- Symbolic Math Toolbox
- Statistics and Machine Learning Toolbox

