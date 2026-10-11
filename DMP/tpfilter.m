function [logl,a,P,yhat,dmpcoefmat] = tpfilter(parvec,Y)
% (c) Eva F. Janssens and Sean McCrary
% These codes are part of the demo codes for the paper "Efficient Estimation 
% of Nonlinear DSGE Models" by Janssens & McCrary.

% First-order Taylor Projection and time-varying Kalman filter.
% parvec = [alfa nu beta delta eta rho sigma sigtheta sigwage]

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

H = diag([sigtheta^2,sigwage^2]);
options = optimoptions('fsolve', ...
    'Display','off', ...
    'SpecifyObjectiveGradient',true, ...
    'FunctionTolerance',1e-12, ...
    'StepTolerance',1e-12, ...
    'OptimalityTolerance',1e-12, ...
    'MaxIterations',500, ...
    'MaxFunctionEvaluations',3000);

T = size(Y,1);
a = zeros(T,1);
P = zeros(T,1);
yhat = zeros(T,2);
dmpcoefmat = zeros(T,2);
loglik = zeros(T,1);

ap = 0;
Pp = par.sigz^2;

for t = 1:T
    [q,qz] = TP_DMP(ap,par,1,options);
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
    P(t) = Pf;
    yhat(t,:) = (h+Z*(af-ap))';
    dmpcoefmat(t,:) = [-q/par.alpha,-qz/par.alpha];

    ap = par.rho*af;
    Pp = par.rho^2*Pf+par.sigma^2;
end

logl = sum(loglik);
end
