function [logl,a,yhat] = EKfilter(parvec,Y,Jglobal,Nglobal)
% (c) Eva F. Janssens and Sean McCrary
% These codes are part of the demo codes for the paper "Efficient Estimation 
% of Nonlinear DSGE Models" by Janssens & McCrary.

% Analytic global solution and oracle extended Kalman filter.
% parvec = [alfa nu beta delta eta rho sigma sigtheta sigwage]

if nargin < 3
    Jglobal = 5000;
end
if nargin < 4
    Nglobal = 24;
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
H = diag([sigtheta^2,sigwage^2]);

T = size(Y,1);
a = zeros(T,1);
yhat = zeros(T,2);
loglik = zeros(T,1);

ap = 0;
Pp = par.sigz^2;

for t = 1:T
    [q,qz] = EVAL_GB_DMP(globalDMP,ap);
    [w,wz] = WAGE_DMP(par,ap);

    h = [-q/par.alpha;w];
    Z = [-qz/par.alpha;wz];
    v = Y(t,:)'-h;
    F = Z*Pp*Z'+H;
    K = (Pp*Z')/F;

    af = ap+K*v;
    Pf = (1-K*Z)*Pp*(1-K*Z)'+K*H*K';

    L = chol(F,'lower');
    e = L\v;
    loglik(t) = -log(2*pi)-sum(log(diag(L)))-0.5*(e'*e);

    a(t) = af;
    yhat(t,:) = (h+Z*(af-ap))';

    ap = par.rho*af;
    Pp = par.rho^2*Pf+par.sigma^2;
end

logl = sum(loglik);
end
