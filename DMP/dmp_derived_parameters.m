function par = dmp_derived_parameters(par)
% (c) Eva F. Janssens and Sean McCrary
% These codes are part of the demo codes for the paper "Efficient Estimation 
% of Nonlinear DSGE Models" by Janssens & McCrary.

% Derived parameters. Call after changing any primitive parameter.

par.gamma     = par.beta*(1-par.delta);
par.kappa     = (1-par.gamma)/(1-par.nu);
par.sigz      = par.sigma/sqrt(1-par.rho^2);
par.bss_q     = -par.kappa/(1-par.gamma*par.rho);
par.bss_theta = -par.bss_q/par.alpha;
par.wage_ss   = par.eta+(1-par.eta)*par.nu;
end
