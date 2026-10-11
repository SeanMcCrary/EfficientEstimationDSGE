# DMP Filtering Demo

This directory contains MATLAB demo code for **“Efficient Estimation of Nonlinear DSGE Models”** by Eva F. Janssens and Sean McCrary.

The code illustrates alternative solution and filtering methods in a Diamond–Mortensen–Pissarides (DMP) search-and-matching model. A single synthetic dataset is generated from the global nonlinear solution and then filtered using several solution/filter combinations.

## Methods Compared

The demo compares five methods:

1. **Log-linear + KF**  
   A log-linear solution combined with the standard Kalman filter.

2. **TP1 + TVKF**  
   First-order Taylor Projection combined with the time-varying Kalman filter.

3. **DP1 + TVKF**  
   First-order Dynamic Perturbation combined with the time-varying Kalman filter.

4. **Global + EKF**  
   The global nonlinear policy function combined with an Extended Kalman Filter.

5. **Global + PF**  
   The global nonlinear policy function combined with a particle filter.

The global solution is used as the data-generating process, so the true latent productivity state and model-implied observables are known. This makes it possible to compare the filtered paths directly with the truth.

## Quick Start

Run

```matlab
demo
```

from the main directory.

The script automatically adds the `helpers/` directory to the MATLAB path.

The demo:

1. loads the baseline DMP calibration;
2. computes the global nonlinear solution;
3. simulates one dataset from the global model;
4. adds measurement error to market tightness and wages;
5. filters the same dataset using all five methods;
6. plots the true and filtered paths for productivity, market tightness, and wages; and
7. reports normalized mean squared error and runtime for each method.

## Output

The main figure contains three panels:

- latent productivity \(z_t\);
- market tightness \(\theta_t\); and
- wages \(w_t\).

Each panel compares the true simulated path with the corresponding filtered paths.

The command window also prints a compact comparison table:

```text
Filtering comparison (based on a single run)
--------------------------------------------------------------------------
Method                 MSE theta    MSE wage       MSE z     Seconds
--------------------------------------------------------------------------
Log-linear + KF             ...         ...         ...         ...
TP1 + TVKF                  ...         ...         ...         ...
DP1 + TVKF                  ...         ...         ...         ...
Global + EKF                ...         ...         ...         ...
Global + PF                 ...         ...         ...         ...
--------------------------------------------------------------------------
```

The reported MSEs are normalized by the unconditional variance of the corresponding variable and multiplied by 100.

Setting

```matlab
save_outputs = true;
```

in `demo.m` saves the figure and summary table to an `output/` directory.

## Files

### Main Demo

- `demo.m`  
  Main entry point. Simulates one dataset from the global solution, applies all five filtering methods, plots the true and filtered paths, and reports filtering accuracy and runtime.

### Calibration

- `dmp_parameters.m`  
  Contains the baseline monthly DMP calibration.

- `dmp_derived_parameters.m`  
  Computes parameters implied by the primitive calibration.

### Global Solution and Simulation

- `GB_DMP.m`  
  Computes the global nonlinear solution of the DMP model.

- `EVAL_GB_DMP.m`  
  Evaluates the global policy function and its derivatives.

- `sim_global_dmp.m`  
  Simulates productivity and evaluates the global policy functions to generate market tightness and wages.

- `WAGE_DMP.m`  
  Evaluates the normalized log-wage policy and its derivatives.

### Filtering Methods

- `kfilter.m`  
  Implements the log-linear solution and standard Kalman filter.

- `tpfilter.m`  
  Implements first-order Taylor Projection combined with the time-varying Kalman filter.

- `dpfilter.m`  
  Implements first-order Dynamic Perturbation combined with the time-varying Kalman filter.

- `EKfilter.m`  
  Implements the global nonlinear solution combined with the Extended Kalman Filter.

- `pfilter.m`  
  Implements the global nonlinear solution combined with a bootstrap particle filter.

### Local Solution Methods

- `TP_DMP.m`  
  Computes the first-order Taylor Projection solution used by `tpfilter.m`.

- `DP_DMP.m`  
  Computes the first-order Dynamic Perturbation solution used by `dpfilter.m`.

### Particle-Filter Utility

- `multinomial_resample.m`  
  Performs multinomial resampling for the bootstrap particle filter.

## Requirements

- MATLAB (tested with R2025b)
- Symbolic Math Toolbox
- Statistics and Machine Learning Toolbox

## Authors

Eva F. Janssens  
Sean McCrary

For questions or issues with the demo code, please contact the authors.
