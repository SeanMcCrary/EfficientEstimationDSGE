function out = nkdmp_zlb_dp_coef(s0, params, N)
%--------------------------------------------------------------------------
% Dynamic Perturbation coefficient recursion for NK-DMP with an
% occasionally binding ZLB.
%
% Uses symbolic files:
%   DP_NKDMP_zlb_slack_Res_eval(xU,s,sp,b,p)
%   DP_NKDMP_zlb_slack_Jac_eval(xU,s,sp,b,p)
%   DP_NKDMP_zlb_slack_IFT_coefficients(xU,s,sp,b,p)
%
%   DP_NKDMP_zlb_bindN_Res_eval(xCN,s,sp,b,p)
%   DP_NKDMP_zlb_bindN_Jac_eval(xCN,s,sp,b,p)
%   DP_NKDMP_zlb_bind_IFT_coefficients(nc,s,sp,b,p)
%
% Inputs
%   s0     : current state [n_{-1}, z, d, m, s, a]' (6x1)
%   params : parameter struct
%   N      : auxiliary path length (positive integer)
%
% Output
%   out.b0U         : 1x28 unconstrained coefficient row at initial point
%   out.b0C         : 1x28 constrained coefficient row at initial point
%   out.b0          : 1x56 row = [b0U, b0C]
%   out.b0_selected : 1x28 row selected by current shadow-rate test
%   out.pathU       : (N+1)x28 unconstrained coefficient path
%   out.pathC       : (N+1)x28 constrained coefficient path
%   out.path        : (N+1)x56 full path = [pathU, pathC]
%   out.nU_path     : N x 1 current n on slack branch
%   out.piU_path    : N x 1 current pi on slack branch
%   out.nC_path     : N x 1 current n on binding branch
%   out.piC_path    : N x 1 current pi on binding branch (recovered from nC)
%   out.isSlack     : N x 1 logical, selected current regime along path
%   out.ntil        : auxiliary path for endogenous predetermined state
%   out.nlag_path   : lagged-employment path used at each node
%   out.elapsed     : total runtime
%
% Notes
%   - The current slack branch solves one scalar equation F_U(n)=0.
%   - The current binding branch solves one scalar equation F_C(n)=0.
%   - On each branch, q, theta, omega, and pi are recovered explicitly
%     inside the symbolic IFT coefficient generators.
%--------------------------------------------------------------------------

t_start = tic;

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

% Structural parameters
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

if isfield(params,'rbar')
    rbar = params.rbar;
elseif isfield(params,'R')
    rbar = log(params.R);
else
    error('Need params.rbar or params.R for the ZLB threshold.');
end

% Parameter vector must match the symbolic ZLB files:
% p = [rz,sz,rd,sd,rm,sm,rs,ss,ra,sa,tau,phipi,phiy,kappa,beta,alpha,Q,chi,gam,del,Npop,nu,rbar]
pvec = [rz; sz; ...
        rd; sd; ...
        rm; sm; ...
        rs; ss; ...
        ra; sa; ...
        tau; phipi; phiy; kappa; bet; alp; ...
        Q; chi; gam; del; Npop; nu; rbar];

%--------------------------------------------------------------------------
% 3. Terminal linear rules
%
% U branch uses base names or *_U if provided.
% Terminally we impose the same unconstrained log-linear rule on both branches.

% U slopes
thU_n = pick(params, {'theta_n_U','theta_n'});
thU_z = pick(params, {'theta_z_U','theta_z'});
thU_d = pick(params, {'theta_d_U','theta_d'});
thU_m = pick(params, {'theta_m_U','theta_m'});
thU_s = pick(params, {'theta_s_U','theta_s'});
thU_a = pick(params, {'theta_a_U','theta_a'});

piU_n = pick(params, {'pi_n_U','pi_n'});
piU_z = pick(params, {'pi_z_U','pi_z'});
piU_d = pick(params, {'pi_d_U','pi_d'});
piU_m = pick(params, {'pi_m_U','pi_m'});
piU_s = pick(params, {'pi_s_U','pi_s'});
piU_a = pick(params, {'pi_a_U','pi_a'});

omU_n = pick(params, {'omega_n_U','omega_n'});
omU_z = pick(params, {'omega_z_U','omega_z'});
omU_d = pick(params, {'omega_d_U','omega_d'});
omU_m = pick(params, {'omega_m_U','omega_m'});
omU_s = pick(params, {'omega_s_U','omega_s'});
omU_a = pick(params, {'omega_a_U','omega_a'});

nU_n  = pick(params, {'n_n_U','n_n'});
nU_z  = pick(params, {'n_z_U','n_z'});
nU_d  = pick(params, {'n_d_U','n_d'});
nU_m  = pick(params, {'n_m_U','n_m'});
nU_s  = pick(params, {'n_s_U','n_s'});
nU_a  = pick(params, {'n_a_U','n_a'});

% U intercepts
thU_0 = pick_or(params, {'theta0_U','theta0'}, 0);
piU_0 = pick_or(params, {'pi0_U','pi0'}, 0);
omU_0 = pick_or(params, {'omega0_U','omega0'}, 0);
nU_0  = pick_or(params, {'n0_U','n0'}, 0);

%--------------------------------------------------------------------------
% 4. Auxiliary exogenous paths, k = 1,...,N+1 for t,...,t+N

ztil = z0    * (rz.^(0:N))';
dtil = d0    * (rd.^(0:N))';
mtil = m0    * (rm.^(0:N))';
stil = s_sh0 * (rs.^(0:N))';
atil = a0    * (ra.^(0:N))';

%--------------------------------------------------------------------------
% 5. Auxiliary path for endogenous predetermined state

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

nlag_path = zeros(N+1,1);
nlag_path(1) = nlag0;
for k = 2:N+1
    nlag_path(k) = ntil(k-1);
end

%--------------------------------------------------------------------------
% 6. Storage

pathU = zeros(N+1,28);
pathC = zeros(N+1,28);

nU_path  = zeros(N,1);
piU_path = zeros(N,1);

nC_path  = zeros(N,1);
piC_path = zeros(N,1);

isSlack = false(N,1);

convSlack = false(N,1);
convBindN = false(N,1);

iterSlack = zeros(N,1);
iterBindN = zeros(N,1);

resSlack = nan(N,1);
resBindN = nan(N,1);

%--------------------------------------------------------------------------
% 7. Terminal coefficients at node k = N+1
%
% Center around ntil(N). Terminally impose the same unconstrained rule on
% both branches.

sN = [nlag_path(N+1); ztil(N+1); dtil(N+1); mtil(N+1); stil(N+1); atil(N+1)];
ntil_center = ntil(N);

pathU(N+1,:) = build_block( ...
    thU_0, [thU_n thU_z thU_d thU_m thU_s thU_a], ...
    piU_0, [piU_n piU_z piU_d piU_m piU_s piU_a], ...
    omU_0, [omU_n omU_z omU_d omU_m omU_s omU_a], ...
    nU_0,  [nU_n  nU_z  nU_d  nU_m  nU_s  nU_a ], ...
    ntil_center, sN);

pathC(N+1,:) = pathU(N+1,:);

% Initial guesses for scalar Newton steps
nU_guess = ntil(N);
nC_guess = ntil(N);

%--------------------------------------------------------------------------
% 8. Backward recursion

for ni = N:-1:1

    % Current node state at time t+(ni-1)
    si = [nlag_path(ni); ztil(ni); dtil(ni); mtil(ni); stil(ni); atil(ni)];

    % Auxiliary object vector:
    % sp = [z~_{t+1}, d~_{t+1}, m~_{t+1}, s~_{t+1}, a~_{t+1}, n~_t]
    spi = [ztil(ni+1); dtil(ni+1); mtil(ni+1); stil(ni+1); atil(ni+1); ntil(ni)];

    % Future regime blocks from next node
    bvec = [pathU(ni+1,:), pathC(ni+1,:)].';

    %--------------------------------------------------------------
    % 8a. Current slack branch: solve nU only
    [nU_star, FU, itU, okU] = DPsolve_n_NKDMP_ZLB_slack(nU_guess, si, spi, bvec, pvec);

    % if ~okU
    %     error('Slack-branch solver failed to converge at node %d. Residual = %g', ni, FU);
    % end

    BU = double(DP_NKDMP_zlb_slack_IFT_coefficients(nU_star, si, spi, bvec, pvec));

    pathU(ni,:)   = real(BU(:).');
    nU_path(ni)   = nU_star;
    piU_path(ni)  = BU(8);

    convSlack(ni) = okU;
    iterSlack(ni) = itU;
    resSlack(ni)  = abs(FU);

    %--------------------------------------------------------------
    % 8b. Current binding branch: solve nC only
    [nC_star, FN, itN, okN] = DPsolve_n_NKDMP_ZLB_bind(nC_guess, si, spi, bvec, pvec);

    % if ~okN
    %     error('Binding-branch solver failed to converge at node %d. Residual = %g', ni, FN);
    % end

    BC = double(DP_NKDMP_zlb_bind_IFT_coefficients(nC_star, si, spi, bvec, pvec));

    pathC(ni,:)   = real(BC(:).');
    nC_path(ni)   = nC_star;
    piC_path(ni)  = BC(8);

    convBindN(ni) = okN;
    iterBindN(ni) = itN;
    resBindN(ni)  = abs(FN);

    %--------------------------------------------------------------
    % 8c. Current regime selection by current shadow Taylor rule
    rhatU = phipi*piU_path(ni) + phiy*(nU_path(ni) + si(2)) + si(4);
    isSlack(ni) = (rhatU > -rbar);

    % Warm starts
    nU_guess = nU_star;
    nC_guess = nC_star;

end

%--------------------------------------------------------------------------
% 9. Output

b0U = pathU(1,:);
b0C = pathC(1,:);

if isSlack(1)
    b0_selected = b0U;
else
    b0_selected = b0C;
end

out = struct();
out.b0U         = b0U;
out.b0C         = b0C;
out.b0          = [b0U, b0C];
out.b0_selected = b0_selected;

out.pathU       = pathU;
out.pathC       = pathC;
out.path        = [pathU, pathC];

out.nU_path     = nU_path;
out.piU_path    = piU_path;
out.nC_path     = nC_path;
out.piC_path    = piC_path;
out.isSlack     = isSlack;

out.ntil        = ntil;
out.nlag_path   = nlag_path;

out.convSlack   = convSlack;
out.convBindN   = convBindN;

out.iterSlack   = iterSlack;
out.iterBindN   = iterBindN;

out.resSlack    = resSlack;
out.resBindN    = resBindN;

out.elapsed     = toc(t_start);

end

%==========================================================================
% Helper: build one 28-coefficient block in order [theta, pi, omega, n]
%==========================================================================
function blk = build_block(th0,thS,pi0,piS,om0,omS,n0,nS,ntil_center,sN)

    z = sN(2); d = sN(3); m = sN(4); s = sN(5); a = sN(6);

    thetaN = th0 + thS(1)*ntil_center + thS(2)*z + thS(3)*d + thS(4)*m + thS(5)*s + thS(6)*a;
    piN    = pi0 + piS(1)*ntil_center + piS(2)*z + piS(3)*d + piS(4)*m + piS(5)*s + piS(6)*a;
    omegaN = om0 + omS(1)*ntil_center + omS(2)*z + omS(3)*d + omS(4)*m + omS(5)*s + omS(6)*a;
    nN     = n0  + nS(1)*ntil_center + nS(2)*z + nS(3)*d + nS(4)*m + nS(5)*s + nS(6)*a;

    blk = [ ...
        thetaN, thS, ...
        piN,    piS, ...
        omegaN, omS, ...
        nN,     nS ];
end

%==========================================================================
% Helper: pick required field from aliases
%==========================================================================
function val = pick(s, names)
    for i = 1:numel(names)
        if isfield(s, names{i})
            val = s.(names{i});
            return
        end
    end
    error('Missing required parameter field. Tried: %s', strjoin(names, ', '));
end

%==========================================================================
% Helper: pick optional field from aliases with fallback default
%==========================================================================
function val = pick_or(s, names, default)
    for i = 1:numel(names)
        if isfield(s, names{i})
            val = s.(names{i});
            return
        end
    end
    val = default;
end