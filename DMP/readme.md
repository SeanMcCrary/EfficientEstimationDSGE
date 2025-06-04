# DMP Model Filtering Comparison

This repository contains MATLAB code to compare our proposed **Time-Varying Kalman Filter (TV-KF)** with three existing filtering methods applied to the **Diamond-Mortensen-Pissarides (DMP)** model:

1. Extended Kalman Filter (EKF)  
2. Log-linear solution and Kalman Filter (KF)  
3. Particle Filter (PF)  
4. Time-Varying Kalman Filter (TV-KF)

## Overview

- The DMP model features AR(1) productivity and is solved globally using a projection method.  
- Simulated data from the global solution is used as the data-generating process (DGP).  
- Each filtering method is applied to this synthetic dataset to infer latent productivity.

## Files

- `demo_filter_comparison.m`  
  Runs the full simulation and applies all four filters.

- `dmp_global_coef.m`, `dmp_resid_global.m`, `sim_global_dmp.m`  
  Solve and simulate the global solution of the DMP model.

- `kfilter_TG.m`, `pfilter_TG.m`, `tpfilter_TG.m`, `EFfilter_TG.m`  
  Implement filtering of latent productivity from observable market tightness using KF, PF, TV-KF, and EKF respectively.

- `Monomials_2.m`, `dmp_tpcoef_sym.m`  
  Auxiliary functions for computing integration nodes/weights and Taylor projection coefficients.

## Requirements

- MATLAB (tested with R2024a)  
- Symbolic Math Toolbox  
- Statistics and Machine Learning Toolbox
