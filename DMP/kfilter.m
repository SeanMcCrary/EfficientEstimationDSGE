function [logl,a,P,yhat] = kfilter(parvec,Y)
% (c) Eva F. Janssens and Sean McCrary
% These codes are part of the demo codes for the paper "Efficient Estimation 
% of Nonlinear DSGE Models" by Janssens & McCrary.

% Log-linear solution and Kalman filter.
% parvec = [alfa nu beta delta eta rho sigma sigtheta sigwage]

alfa    = parvec(1);
nu      = parvec(2);
beta    = parvec(3);
delta   = parvec(4);
eta     = parvec(5);
rho     = parvec(6);
sigma   = parvec(7);
sigtheta = parvec(8);
sigwage  = parvec(9);

gamma = beta*(1-delta);
kappa = (1-gamma)/(1-nu);
sigz  = sigma/sqrt(1-rho^2);

theta_z = kappa/(alfa*(1-gamma*rho));
wage_z  = eta/(eta+(1-eta)*nu);
Z = [theta_z;wage_z];
H = diag([sigtheta^2,sigwage^2]);

T = size(Y,1);
a = zeros(T,1);
P = zeros(T,1);
yhat = zeros(T,2);
loglik = zeros(T,1);

ap = 0;
Pp = sigz^2;

for t = 1:T
    v = Y(t,:)'-Z*ap;
    F = Z*Pp*Z'+H;
    K = (Pp*Z')/F;

    af = ap+K*v;
    Pf = (1-K*Z)*Pp*(1-K*Z)'+K*H*K';

    L = chol(F,'lower');
    e = L\v;
    loglik(t) = -log(2*pi)-sum(log(diag(L)))-0.5*(e'*e);

    a(t) = af;
    P(t) = Pf;
    yhat(t,:) = (Z*af)';

    ap = rho*af;
    Pp = rho^2*Pf+sigma^2;
end

logl = sum(loglik);
end
