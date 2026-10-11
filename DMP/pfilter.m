function [logl,a,yhat] = pfilter(parvec,Y,M,Jglobal,Nglobal,seed)
% (c) Eva F. Janssens and Sean McCrary
% These codes are part of the demo codes for the paper "Efficient Estimation 
% of Nonlinear DSGE Models" by Janssens & McCrary.

% Analytic global solution and bootstrap particle filter.
% parvec = [alfa nu beta delta eta rho sigma sigtheta sigwage]

if nargin < 4
    Jglobal = 5000;
end
if nargin < 5
    Nglobal = 24;
end
if nargin < 6
    seed = 1;
end

par.alpha = parvec(1);
par.nu    = parvec(2);
par.beta  = parvec(3);
par.delta = parvec(4);
par.eta   = parvec(5);
par.rho   = parvec(6);
par.sigma = parvec(7);
sigtheta  = parvec(8);
sigwage   = parvec(9);
par = dmp_derived_parameters(par);

globalDMP = GB_DMP(par,Jglobal,Nglobal);
T = size(Y,1);

old_rng = rng;
rng(seed,'twister');
shock = randn(T,M);
resampling_uniform = rand(T,M);
rng(old_rng);

particle = par.sigz*shock(1,:)';
weight = ones(M,1)/M;
a = zeros(T,1);
yhat = zeros(T,2);
loglik = zeros(T,1);
log_constant = -log(2*pi)-log(sigtheta)-log(sigwage);

for t = 1:T
    if t > 1
        particle = par.rho*particle+par.sigma*shock(t,:)';
    end

    q = EVAL_GB_DMP(globalDMP,particle);
    theta = -q/par.alpha;
    wage = WAGE_DMP(par,particle);

    error_theta = (Y(t,1)-theta)/sigtheta;
    error_wage  = (Y(t,2)-wage)/sigwage;
    log_density = log_constant-0.5*(error_theta.^2+error_wage.^2);
    log_weight = log(weight)+log_density;

    largest = max(log_weight);
    loglik(t) = largest+log(sum(exp(log_weight-largest)));
    weight = exp(log_weight-loglik(t));

    a(t) = sum(weight.*particle);
    yhat(t,:) = [sum(weight.*theta),sum(weight.*wage)];

    if 1/sum(weight.^2) < 0.25*M
        index = multinomial_resample(weight,resampling_uniform(t,:));
        particle = particle(index);
        weight = ones(M,1)/M;
    end
end

logl = sum(loglik);
end
