function [q,qz,qzz] = TP_DMP(z,par,order,options)
% (c) Eva F. Janssens and Sean McCrary
% These codes are part of the demo codes for the paper "Efficient Estimation 
% of Nonlinear DSGE Models" by Janssens & McCrary.

% First- or second-order Taylor Projection evaluated at z.

if nargin < 3
    order = 1;
end
if nargin < 4
    options = optimoptions('fsolve', ...
        'Display','off', ...
        'SpecifyObjectiveGradient',true, ...
        'FunctionTolerance',1e-12, ...
        'StepTolerance',1e-12, ...
        'OptimalityTolerance',1e-12, ...
        'MaxIterations',500, ...
        'MaxFunctionEvaluations',3000);
end

zsize = size(z);
z = z(:);
q   = zeros(size(z));
qz  = zeros(size(z));
qzz = zeros(size(z));

for iz = 1:numel(z)
    z0 = z(iz);
    Ez = exp(z0);

    if order == 1
        slope = fsolve(@(s) TP1_residual(s,z0,par), ...
            par.bss_q,options);

        C = slope*(1-par.rho)*z0+0.5*slope^2*par.sigma^2;
        denominator = 1-par.gamma*exp(C);

        q(iz)  = log(denominator)-log(par.kappa*(Ez-par.nu));
        qz(iz) = slope;
    else
        coefficient = fsolve(@(x) TP2_residual(x,z0,par), ...
            [par.bss_q;0],options);

        slope = coefficient(1);
        curvature = coefficient(2);
        mu = (par.rho-1)*z0;
        S = 1+par.sigma^2*curvature;
        B = slope+curvature*mu;
        T = exp(-0.5*log(S)-slope*mu-0.5*curvature*mu^2 ...
            +0.5*par.sigma^2*B^2/S);
        denominator = 1-par.gamma*T;

        q(iz)   = log(denominator)-log(par.kappa*(Ez-par.nu));
        qz(iz)  = slope;
        qzz(iz) = curvature;
    end
end

q   = reshape(q,zsize);
qz  = reshape(qz,zsize);
qzz = reshape(qzz,zsize);
end

function [r,J] = TP1_residual(s,z,par)
Ez = exp(z);
C = s*(1-par.rho)*z+0.5*s^2*par.sigma^2;
S = exp(C);

r = s*(Ez-par.nu)*(1-par.gamma*par.rho*S) ...
    +Ez*(1-par.gamma*S);

if nargout > 1
    H = (1-par.rho)*z+s*par.sigma^2;
    J = (Ez-par.nu)*(1-par.gamma*par.rho*S) ...
        -par.gamma*S*H*(par.rho*s*(Ez-par.nu)+Ez);
end
end

function [r,J] = TP2_residual(x,z,par)
s = x(1);
c = x(2);
Ez = exp(z);
mu = (par.rho-1)*z;
v = par.sigma^2;
S = 1+v*c;

if S <= 0
    r = 1e10*ones(2,1);
    J = zeros(2,2);
    return
end

B = s+c*mu;
T = exp(-0.5*log(S)-s*mu-0.5*c*mu^2+0.5*v*B^2/S);
D = 1-par.gamma*T;

if D <= 0
    r = 1e10*ones(2,1);
    J = zeros(2,2);
    return
end

H = Ez/(Ez-par.nu);
U = B/S;
V = B^2/S^2-c/S;

r = [ ...
    -s-H*D+par.gamma*par.rho*T*U; ...
    s^2-c-H*D-par.gamma*par.rho^2*T*V];

if nargout > 1
    G = mu-v*s;
    Ls = -G/S;
    Lc = -0.5*(v*S+G^2)/S^2;
    Us = 1/S;
    Uc = G/S^2;
    Vs = 2*B/S^2;
    Vc = 2*B*G/S^3-1/S^2;

    J = zeros(2,2);
    J(1,1) = -1+H*par.gamma*T*Ls ...
        +par.gamma*par.rho*T*(Ls*U+Us);
    J(1,2) = H*par.gamma*T*Lc ...
        +par.gamma*par.rho*T*(Lc*U+Uc);
    J(2,1) = 2*s+H*par.gamma*T*Ls ...
        -par.gamma*par.rho^2*T*(Ls*V+Vs);
    J(2,2) = -1+H*par.gamma*T*Lc ...
        -par.gamma*par.rho^2*T*(Lc*V+Vc);
end
end
