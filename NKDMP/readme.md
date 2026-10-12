# NKDMP Filtering Demo

This directory contains MATLAB demo code for **“Efficient Estimation of Nonlinear DSGE Models”** by Eva F. Janssens and Sean McCrary.

The demo illustrates filtering in a New Keynesian Diamond–Mortensen–Pissarides (NKDMP) model using two empirical samples. All parameters are fixed; the demo does not perform estimation.

## Samples and Methods

| Data file | Sample | Methods |
| --- | --- | --- |
| `demo_data_prezlb.mat` | 1984Q1–2007Q4 | Log-linear + KF, DP + TVKF, TP + TVKF |
| `demo_data_zlb.mat` | 1984Q1–2019Q4 | DP-ZLB + TVKF |

KF denotes the Kalman filter, TVKF the time-varying Kalman filter, DP Dynamic Perturbation, and TP Taylor Projection. DP-ZLB incorporates the zero lower bound on the nominal interest rate.

The three pre-ZLB methods use the same fixed parameter vector. The sample including the ZLB uses a separate fixed parameter vector from the saved full-sample results. Both parameter vectors are included in the data files.

## Quick Start

Run

```matlab
demo
```

from the main directory. The script automatically adds the `helpers/` directory and its subdirectories to the MATLAB path.

## Data

Each data file contains quarterly dates and seven observed series, used in the following order:

1. GDP deflator inflation (`pi_obs`)
2. CPI inflation (`pi_obs2`)
3. Labor input, measured using hours (`n_obs`)
4. Market tightness (`tg_obs`)
5. Nominal interest rate (`r_obs`)
6. Separation rate (`s_obs`)
7. Job-finding rate (`jf_obs`)

## Output

The demo produces two figures, each showing observed and filtered values for all seven variables:

- **Pre-ZLB sample:** log-linear, DP, and TP.
- **Sample including the ZLB:** DP-ZLB.

The figures display the series in 100 times their input log units. Filtered observables and state estimates remain available in the MATLAB workspace.

## Requirements

- MATLAB R2025b
- Statistics and Machine Learning Toolbox
