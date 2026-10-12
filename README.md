# EfficientEstimationDSGE

This repository contains MATLAB code accompanying the paper **“Efficient Estimation of Nonlinear DSGE Models”** by **Eva F. Janssens** and **Sean McCrary**.

The demos illustrate solution and filtering methods for nonlinear DSGE models using simulated and empirical data. They include Taylor Projection (TP) and Dynamic Perturbation (DP) combined with the time-varying Kalman filter (TVKF), alongside log-linear and global-solution benchmarks.

Each model directory contains a `demo.m` script and a separate README with further details. These are demonstrations of the methodology proposed in the paper, not replication files.

## Contents

### [DMP](DMP)

A filtering comparison in the **Diamond–Mortensen–Pissarides (DMP)** search-and-matching model.

One dataset is simulated from the global nonlinear solution and filtered using five methods:

- Log-linear solution + Kalman filter (KF)
- First-order Taylor Projection (TP1) + TVKF
- First-order Dynamic Perturbation (DP1) + TVKF
- Global solution + Extended Kalman Filter (EKF)
- Global solution + particle filter (PF)

The demo plots true and filtered productivity, market tightness, and wages, and reports normalized mean squared errors and runtimes.

### [TextbookNewKeynesian](TextbookNewKeynesian)

A filtering illustration in the **standard three-equation New Keynesian model**, using quarterly U.S. output, inflation, and nominal interest-rate data from **1966Q1–2007Q4**.

The demo loads `demo_data.mat` and applies four methods at a common fixed calibration: **log-linear + KF, TP1 + TVKF, DP1 + TVKF, and DP1-ZLB + TVKF**. The observed zero lower bound does not bind in this sample.

It plots the filtered latent states and observed versus filtered variables, and reports log likelihoods and runtimes.

### [NKDMP](NKDMP)

A filtering illustration in the **New Keynesian Diamond–Mortensen–Pissarides (NKDMP)** model using two empirical samples:

| Data file | Sample | Methods |
| --- | --- | --- |
| `demo_data_prezlb.mat` | 1984Q1–2007Q4 | Log-linear + KF, DP + TVKF, TP + TVKF |
| `demo_data_zlb.mat` | 1984Q1–2019Q4 | DP-ZLB + TVKF |

Both files contain seven observed series: GDP deflator inflation, CPI inflation, labor input measured using hours, market tightness, the nominal interest rate, the separation rate, and the job-finding rate.

The demo uses fixed parameter values included in the data files and produces two figures comparing observed and filtered values of all seven variables. The three pre-ZLB methods share one parameter vector; the full sample uses a separate parameter vector. No estimation is performed.

### [StochasticGrowth_RegimeSwitching](StochasticGrowth_RegimeSwitching)

A **stochastic growth model with regime switching** in productivity persistence and volatility.

The demo uses simulated data to illustrate filtering of latent states and regime probabilities. It combines regime-dependent Taylor Projection solutions with the Hamilton filter.

## Quick Start

Open the directory for the model you want to run and execute:

```matlab
demo
```

See the README in that directory for details on its data, methods, and output.

## Requirements

- MATLAB R2025b
- Optimization Toolbox for the DMP Taylor Projection solver
- Statistics and Machine Learning Toolbox for demos using statistical functions
- Symbolic Math Toolbox for generating symbolic model equations

The textbook New Keynesian filtering demo runs in base MATLAB. Toolbox requirements vary across the other examples.

## Contact

This is a work in progress. Code will be updated periodically.

For questions, suggestions, or concerns, contact: mccrary.65 **at** osu.edu
