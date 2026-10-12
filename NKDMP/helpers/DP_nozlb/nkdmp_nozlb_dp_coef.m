function out = nkdmp_nozlb_dp_coef(s0, params, N)
%--------------------------------------------------------------------------
% Dynamic Perturbation (no ZLB) coefficient recursion for NK-DMP.
%
% Uses symbolic files:
%   DP_NKDMP_nozlb_Res_eval(x,s,sp,b,p)
%   DP_NKDMP_nozlb_Jac_eval(x,s,sp,b,p)
%   DP_NKDMP_nozlb_IFT_coefficients(x,s,sp,b,p)
%
% Inputs
%   s0     : current state [n_{-1}, z, d, m, s, a]' (6x1)
%   params : parameter struct
%   N      : auxiliary path length (positive integer)
%
% Output
%   out.b0         : 1x28 coefficient row at the initial point
%   out.path       : (N+1)x28 coefficient rows along the auxiliary path
%   out.pi_path    : N x 1 solved scalar pi values from backward recursion
%   out.ntil       : auxiliary path for endogenous predetermined state
%   out.nlag_path  : lagged-employment path used at each node
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% 0. Basic checks

s0 = s0(:);

if numel(s0) ~= 6
    error('s0 must be 6x1.');
end
if ~(isscalar(N) && N >= 1 && N == floor(N))
    error('N must be a positive integer.');
end

%--------------------------------------------------------------------------
% 1. Current state

nlag0 = s0(1);
z0    = s0(2);
d0    = s0(3);
m0    = s0(4);
s_sh0 = s0(5);
a0    = s0(6);

%--------------------------------------------------------------------------
% 2. Parameters

% AR coefficients
rz = params.rho_z;  rd = params.rho_d;  rm = params.rho_m;
rs = params.rho_s;  ra = params.rho_a;

% Shock scales
sz = params.sig_z;  sd = params.sig_d;  sm = params.sig_m;
ss = params.sig_s;  sa = params.sig_a;

% Preferences / policy / technology
tau   = params.tau;
phipi = params.phi_pi;
phiy  = params.phi_y;
bet   = params.beta;
alp   = params.alpha;

Q     = params.Q;
gam   = params.gamma;
del   = params.delta;
Npop  = params.N;
nu    = params.nu;

% Use kappa and chi directly if available; otherwise construct them once
if isfield(params,'kappa')
    kappa = params.kappa;
else
    kappa = 1/(params.phi * params.Pi^2);
end

if isfield(params,'chi')
    chi = params.chi;
elseif isfield(params,'c') && isfield(params,'eta')
    chi = Q*(1-params.eta)/params.c;
else
    chi = (1 - bet*(1-del)) / (1 - gam - nu);
end

% Parameter vector MUST match the updated symbolic files
pvec = [rz; sz; ...
        rd; sd; ...
        rm; sm; ...
        rs; ss; ...
        ra; sa; ...
        tau; phipi; phiy; kappa; bet; alp; ...
        Q; chi; gam; del; Npop; nu];

%--------------------------------------------------------------------------
% 3. Terminal linear rule coefficients
%
% Expected interpretation:
%   y = y0 + y_n*nlag + y_z*z + y_d*d + y_m*m + y_s*s + y_a*a
%
% These are used only to initialize the terminal future rule.

% theta slopes
th_n = params.theta_n; th_z = params.theta_z; th_d = params.theta_d;
th_m = params.theta_m; th_s = params.theta_s; th_a = params.theta_a;

% pi slopes
pi_n = params.pi_n; pi_z = params.pi_z; pi_d = params.pi_d;
pi_m = params.pi_m; pi_s = params.pi_s; pi_a = params.pi_a;

% omega slopes
om_n = params.omega_n; om_z = params.omega_z; om_d = params.omega_d;
om_m = params.omega_m; om_s = params.omega_s; om_a = params.omega_a;

% n slopes
n_n  = params.n_n; n_z = params.n_z; n_d = params.n_d;
n_m  = params.n_m; n_s = params.n_s; n_a = params.n_a;

% optional intercepts
if isfield(params,'theta0'); theta0_bar = params.theta0; else; theta0_bar = 0; end
if isfield(params,'pi0');    pi0_bar    = params.pi0;    else; pi0_bar    = 0; end
if isfield(params,'omega0'); omega0_bar = params.omega0; else; omega0_bar = 0; end
if isfield(params,'n0');     n0_bar     = params.n0;     else; n0_bar     = 0; end

%--------------------------------------------------------------------------
% 4. Auxiliary exogenous paths, indexed k = 1,...,N+1 for t,...,t+N

ztil = z0    * (rz.^(0:N))';
dtil = d0    * (rd.^(0:N))';
mtil = m0    * (rm.^(0:N))';
stil = s_sh0 * (rs.^(0:N))';
atil = a0    * (ra.^(0:N))';

%--------------------------------------------------------------------------
% 5. Auxiliary path for endogenous predetermined state
%
% Baseline rule:
%   \tilde n_{t+i} = b40 + b41 \tilde n_{t+i-1} + b42 \tilde z_{t+i} + ...
% with \tilde n_{t-1} = nlag0.

b40 = params.nbar0;
b41 = params.nbar1;
b42 = params.nbar2;
b43 = params.nbar3;
b44 = params.nbar4;
b45 = params.nbar5;
b46 = params.nbar6;

ntil = zeros(N+1,1);
ntil(1) = b40 + b41*nlag0   + b42*ztil(1) + b43*dtil(1) + b44*mtil(1) + b45*stil(1) + b46*atil(1);

for k = 2:N+1
    ntil(k) = b40 + b41*ntil(k-1) + b42*ztil(k) + b43*dtil(k) + b44*mtil(k) + b45*stil(k) + b46*atil(k);
end

% lagged employment entering each node k
nlag_path = zeros(N+1,1);
nlag_path(1) = nlag0;
for k = 2:N+1
    nlag_path(k) = ntil(k-1);
end

%--------------------------------------------------------------------------
% 6. Storage

DPcoefss = zeros(N+1,28);
pi_path  = zeros(N,1);

%--------------------------------------------------------------------------
% 7. Terminal coefficients at node k = N+1 (time t+N)
%
% These coefficients will be used as the "future rule" when solving node k=N.
% The future rule is centered around ntil(N), i.e. the auxiliary current n at
% the previous node.

sN = [nlag_path(N+1); ztil(N+1); dtil(N+1); mtil(N+1); stil(N+1); atil(N+1)];

% Centering level for the future rule used by node N
ntil_center = ntil(N);

% Terminal constants = terminal linear rule evaluated at the auxiliary point
thetaN = theta0_bar + th_n*ntil_center + th_z*sN(2) + th_d*sN(3) + th_m*sN(4) + th_s*sN(5) + th_a*sN(6);
piN    = pi0_bar    + pi_n*ntil_center + pi_z*sN(2) + pi_d*sN(3) + pi_m*sN(4) + pi_s*sN(5) + pi_a*sN(6);
omegaN = omega0_bar + om_n*ntil_center + om_z*sN(2) + om_d*sN(3) + om_m*sN(4) + om_s*sN(5) + om_a*sN(6);
nN     = n0_bar     + n_n *ntil_center + n_z *sN(2) + n_d *sN(3) + n_m *sN(4) + n_s *sN(5) + n_a *sN(6);

% Pack terminal coefficients in the SAME order expected by the symbolic DP files:
% [theta_block, pi_block, omega_block, n_block]
DPcoefss(N+1,:) = [ ...
    thetaN, th_n, th_z, th_d, th_m, th_s, th_a, ...
    piN,    pi_n, pi_z, pi_d, pi_m, pi_s, pi_a, ...
    omegaN, om_n, om_z, om_d, om_m, om_s, om_a, ...
    nN,     n_n,  n_z,  n_d,  n_m,  n_s,  n_a ];

% Initial guess for scalar Newton
pi00 = piN;

%--------------------------------------------------------------------------
% 8. Backward recursion

for ni = N:-1:1

    % Current node state at time t+(ni-1)
    si = [nlag_path(ni); ztil(ni); dtil(ni); mtil(ni); stil(ni); atil(ni)];

    % Auxiliary objects for the symbolic functions:
    % sp = [z~_{t+1}, d~_{t+1}, m~_{t+1}, s~_{t+1}, a~_{t+1}, n~_t]
    spi = [ztil(ni+1); dtil(ni+1); mtil(ni+1); stil(ni+1); atil(ni+1); ntil(ni)];

    % Future coefficients from the next auxiliary node
    bvec = DPcoefss(ni+1,:).';

    % Solve scalar pi_t
    [pi_star, ~, ~] = DPsolve_pi_NKDMP_noZLB(pi00, si, spi, bvec, pvec);
    pi_path(ni) = pi_star;

    % Evaluate current coefficients.
    % The updated symbolic generator already returns the coefficients in the
    % SAME order as the input b:
    % [theta_block, pi_block, omega_block, n_block]
    B = DP_NKDMP_nozlb_IFT_coefficients(pi_star, si, spi, bvec, pvec);

    DPcoefss(ni,:) = B(:).';

    % Warm start next Newton solve
    pi00 = pi_star;
end

%--------------------------------------------------------------------------
% 9. Output

out = struct();
out.b0        = DPcoefss(1,:);
out.path      = DPcoefss;
out.pi_path   = pi_path;
out.ntil      = ntil;
out.nlag_path = nlag_path;

end