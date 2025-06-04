# Three-Equation New Keynesian Model

This repository contains MATLAB code for full-information Bayesian estimation of the textbook **New Keynesian** model using a Random-Walk Metropolis-Hastings sampler.

## Overview

- The model features two policy variables—**employment** and **inflation**—and three structural shocks:
  - **Productivity shock**
  - **Discount factor shock**
  - **Monetary policy shock**
- Estimation is conducted on U.S. data for output, inflation, and the federal funds rate from 1966:Q1 to 2007:Q4.
- Policy functions are computed using a Newton solver.

## Files

- `demo_estimation.m`, `demo_data.mat`  
  Main script for running the estimation using the Random-Walk Metropolis-Hastings algorithm. Outputs posterior draws and plots results.

- `tpfilter.m`, `nk_log_tp_coef.m`  
  Functions to filter the data and compute policy function coefficients via a Newton solver.

- `transformParamsInvME.m`, `transformParamsME.m`  
  Functions to transform parameter vectors from the sampler into structural parameters, ensuring appropriate bounds (e.g., positive variances).

## Requirements

- MATLAB (tested with R2024a)
- Symbolic Math Toolbox
- Statistics and Machine Learning Toolbox
