function [coefficients, params] = nk_loglinear_coefficients(params)
% Eva F. Janssens and Sean McCrary
% "Efficient Estimation of Nonlinear DSGE Models"

%NK_LOGLINEAR_COEFFICIENTS Original closed-form log-linear solution.
%
% The formulas are lifted from the original solution workspace. The returned
% parameter fields are the terminal slopes expected by both DP recursions.

tau   = params.tau;
bet   = params.beta;
eta   = params.eta;
kap   = params.kappa;
phipi = params.phi_pi;
phiy  = params.phi_y;
rhoz  = params.rho_z;
rhod  = params.rho_d;
rhom  = params.rho_m;

Del_rhoz = (tau*(1-rhoz)+phiy)*(1-bet*rhoz) ...
    + kap*(eta+tau)*(phipi-rhoz);
Del_rhod = (tau*(1-rhod)+phiy)*(1-bet*rhod) ...
    + kap*(eta+tau)*(phipi-rhod);
Del_rhom = (tau*(1-rhom)+phiy)*(1-bet*rhom) ...
    + kap*(eta+tau)*(phipi-rhom);

n_z = -((tau*(1-rhoz)+phiy)*(1-bet*rhoz) ...
    + kap*(tau-1)*(phipi-rhoz))/Del_rhoz;
n_d = ((1-rhod)*(1-bet*rhod))/Del_rhod;
n_m = -(1-bet*rhom)/Del_rhom;

pi_z = -(kap*(eta+1)*(tau*(1-rhoz)+phiy))/Del_rhoz;
pi_d = kap*(eta+tau)*(1-rhod)/Del_rhod;
pi_m = -kap*(eta+tau)/Del_rhom;

params.n_z  = n_z;
params.n_d  = n_d;
params.n_m  = n_m;
params.pi_z = pi_z;
params.pi_d = pi_d;
params.pi_m = pi_m;

coefficients.n  = [n_z,n_d,n_m];
coefficients.pi = [pi_z,pi_d,pi_m];
coefficients.state_order = {'z','d','m'};
coefficients.Delta = [Del_rhoz,Del_rhod,Del_rhom];
coefficients.vector = [0;pi_z;pi_d;pi_m;0;n_z;n_d;n_m];

end
