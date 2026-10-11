function [w,wz,wzz] = WAGE_DMP(par,z)

% (c) Eva F. Janssens and Sean McCrary
% These codes are part of the demo codes for the paper "Efficient Estimation 
% of Nonlinear DSGE Models" by Janssens & McCrary.

% Exact normalized log-wage policy.

productive = par.eta*exp(z);
wage_level = productive+(1-par.eta)*par.nu;
w   = log(wage_level/par.wage_ss);
wz  = productive./wage_level;
wzz = wz.*(1-wz);
end
