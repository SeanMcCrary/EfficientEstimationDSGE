function par = dmp_parameters()
% (c) Eva F. Janssens and Sean McCrary
% These codes are part of the demo codes for the paper "Efficient Estimation 
% of Nonlinear DSGE Models" by Janssens & McCrary.

% Baseline monthly calibration used in GlobalDMP.m.

par.fss   = 0.50;
par.uss   = 0.10;
par.beta  = 0.96^(1/52);
par.delta = par.fss*par.uss/((1-par.fss)*(1-par.uss));
par.nu    = 0.84;
par.alpha = 0.50;
par.eta   = 0.50;

% Monthly AR(1) recovered from the quarterly Hamilton-filtered moments.
par.rho   = 0.9895; % 0.9786729783573109;
par.sigma = 0.0034; % 0.006410665347092723;

par = dmp_derived_parameters(par);
end
