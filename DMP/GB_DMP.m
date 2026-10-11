function globalDMP = GB_DMP(par,J,N)
% (c) Eva F. Janssens and Sean McCrary
% These codes are part of the demo codes for the paper "Efficient Estimation 
% of Nonlinear DSGE Models" by Janssens & McCrary.

% Analytic global solution truncated at j=0,...,J.
% The order-N power series makes repeated EKF and particle evaluations fast.

j         = 0:J;
rho_j     = par.rho.^j;
gamma_j   = par.gamma.^j;
var_j     = par.sigma^2*(1-par.rho.^(2*j))/(1-par.rho^2);
exp_var_j = exp(0.5*var_j);

a = zeros(N+1,1);
a(1) = par.kappa*sum(gamma_j.*(exp_var_j-par.nu));

rho_power = ones(size(j));
inverse_factorial = 1;
for n = 1:N
    rho_power = rho_power.*rho_j;
    inverse_factorial = inverse_factorial/n;
    a(n+1) = par.kappa*inverse_factorial ...
        *sum(gamma_j.*exp_var_j.*rho_power);
end

globalDMP.a = a;
globalDMP.J = J;
globalDMP.N = N;
end
