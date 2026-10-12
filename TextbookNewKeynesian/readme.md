# Textbook New Keynesian Filtering Demo

This directory contains MATLAB demo code for **“Efficient Estimation of Nonlinear DSGE Models”** by Eva F. Janssens and Sean McCrary.

The demo loads `demo_data.mat` and filters the same empirical observations using four methods at a common fixed calibration.

## Methods Compared

1. **Log-linear + KF**  
   A log-linear solution combined with the standard Kalman filter.

2. **TP1 + TVKF**  
   First-order Taylor Projection combined with the time-varying Kalman filter.

3. **DP1 + TVKF**  
   First-order Dynamic Perturbation combined with the time-varying Kalman filter.

4. **DP1-ZLB + TVKF**  
   First-order Dynamic Perturbation with a zero lower bound combined with the time-varying Kalman filter.

The DP-ZLB filter uses the observed binding indicator, which is false throughout this sample. Its solution still accounts for the possibility of future binding. The other three methods do not impose the lower bound.

The fixed structural parameters, shock processes, measurement errors, and 12-period DP horizon follow `TextbookNK/main_filter_zlb.m`. The filter and solution functions are copied unchanged from TextbookNK and are called directly by the demo.

## Quick Start

Run

```matlab
demo
```

from the main directory. The script automatically adds `helpers/` to the MATLAB path.

The demo loads the data, sets the parameters, runs the four filters, prints likelihoods and runtimes, and plots filtered states and observables.

## Data

`demo_data.mat` contains 168 quarterly observations of U.S. output, inflation, and the nominal interest rate, covering 1966:Q1 through 2007:Q4. It is copied unchanged from the original GitHub demo.

Output enters as supplied. Reference inflation is fixed at the sample mean, and the reference nominal rate is `pi_bar-log(beta)`, consistent with the calibrated discount factor. Inflation and the nominal rate are expressed as deviations from these references for filtering; the offsets are added back for plotting. The lower bound in rate deviations is `-r_bar`. All parameters are set in `demo.m`.

## Output

The command window reports each method's log likelihood and runtime. The figure shows the three filtered latent states and compares observed and filtered output, inflation, and the nominal rate.

Filter results remain available in the workspace as `fits`, with the comparison table in `comparison`.

## Files

- `demo.m`: main filtering demo.
- `demo_data.mat`: original empirical dataset.
- `helpers/nk_filter_tvkf.m`: standard and time-varying Kalman filters.
- `helpers/nk_loglinear_coefficients.m`: log-linear solution.
- `helpers/nk_log_tp_coef.m`: Taylor Projection solution.
- `helpers/nk_nozlb_dp_coef.m` and the four `DP_nozlb_*`/`DPsolve_pi_noZLB.m` helpers: Dynamic Perturbation solution.
- `helpers/nk_zlb_dp_coef.m` and the four `DP_zlb_*`/`DPsolve_pi_ZLB.m` helpers: Dynamic Perturbation solution with the zero lower bound.

## Requirements

- MATLAB (tested with R2025b)

The demo uses base MATLAB and requires no additional toolboxes.

## Authors

Eva F. Janssens  
Sean McCrary
