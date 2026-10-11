function out = sim_global_dmp(par,globalDMP,T,z0,innovations)
% (c) Eva F. Janssens and Sean McCrary
% These codes are part of the demo codes for the paper "Efficient Estimation 
% of Nonlinear DSGE Models" by Janssens & McCrary.

% Simulate productivity and evaluate the analytic global policy.

if nargin < 5
    innovations = randn(T-1,numel(z0));
end

z0 = z0(:)';
z = zeros(T,numel(z0));
z(1,:) = z0;

for t = 2:T
    z(t,:) = par.rho*z(t-1,:)+par.sigma*innovations(t-1,:);
end

q = EVAL_GB_DMP(globalDMP,z);
w = WAGE_DMP(par,z);

out.tightness = -q/par.alpha;
out.wage      = w;
out.logtfp    = z;
out.q         = q;
end
