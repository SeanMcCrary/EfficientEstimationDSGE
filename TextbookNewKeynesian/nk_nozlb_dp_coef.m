function out = nk_nozlb_dp_coef(s,params,N)
% Eva F. Janssens and Sean McCrary
% "Efficient Estimation of Nonlinear DSGE Models"
%--------------------------------------------------------------------------
% Description
% This file constructs policy functions of the 3-equation NK model
% model without a ZLB via Dynamic Perturbation three inputs as arguments
%
% 1. s      - vector of states s_t = [z_t d_t m_t]
% 2. params - structure including model parameters
% 3. N      - integer, length of auxiliary path
%
% The function outputs the policy coefficients for inflation and employment
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
% Step 0. Parameters

% current state
z = s(1); d = s(2); m = s(3);

% exogenous shocks
rhoz = params.rho_z; rhod = params.rho_d; rhom = params.rho_m;
sigz = params.sig_z; sigd = params.sig_d; sigm = params.sig_m;

% economic parameters
tau   = params.tau;
eta   = params.eta;
bet   = params.beta;
kap   = params.kappa;
phiy  = params.phi_y;
phipi = params.phi_pi;

% log-linear policy
n_z  = params.n_z;
n_d  = params.n_d;
n_m  = params.n_m;
pi_z = params.pi_z;
pi_d = params.pi_d;
pi_m = params.pi_m;

% economic parameters
pvec = [rhoz sigz rhod sigd rhom sigm tau phipi phiy kap eta bet]';

%--------------------------------------------------------------------------
% Step 1. Auxiliary Path

zt = z*(rhoz.^(0:N))';
dt = d*(rhod.^(0:N))';
mt = m*(rhom.^(0:N))';

DPcoefss        = zeros(N+1,8);
DPcoefss(N+1,:) = [0 pi_z pi_d pi_m 0 n_z n_d n_m];

% inital guess (log-linear)
pi00 = pi_z.*zt(end) + pi_d.*dt(end) + pi_m.*mt(end);
n00  = n_z.*zt(end)  + n_d.*dt(end)  + n_m.*mt(end);

DPcoefss(N+1,1) = pi00;
DPcoefss(N+1,5) = n00;

%--------------------------------------------------------------------------
% Step 2. Backwards recursion

for ni=N:-1:1

    % current state (z_t+i d_t+i m_t+i)
    si = [zt(ni);dt(ni);mt(ni)];

    % future state (z_t+i+1 d_t+i+1 m_t+i+1)
    spi = [zt(ni+1);dt(ni+1);mt(ni+1)];

    % future policy
    bvec = DPcoefss(ni+1,:)';

    % solve for inflation
    [pi_star, Fval, it, converged] = ...
        DPsolve_pi_noZLB(pi00, si, bvec, pvec);
    if ~converged
        error('SimpleNK:DPNoZLBNewtonFailed', ...
            ['No-ZLB DP Newton solve failed at backward step %d ' ...
             '(iteration %d, residual %.3e).'],ni,it,abs(Fval));
    end

    % policy coeffs from IFT
    Fp = DP_nozlb_Jac_eval(pi_star, si, bvec, pvec);
    if ~isreal(Fp) || ~isfinite(Fp) || abs(Fp) < 1e-6
        error('SimpleNK:DPNoZLBIFTIllConditioned', ...
            'No-ZLB DP IFT is ill-conditioned at backward step %d.',ni);
    end
    B = DP_nozlb_IFT_coefficients(pi_star, si, spi, bvec, pvec);
    if ~isreal(B) || any(~isfinite(B))
        error('SimpleNK:DPNoZLBIFTFailed', ...
            'No-ZLB DP IFT returned a nonfinite coefficient at backward step %d.',ni);
    end

    % update policy along the path
    DPcoefss(ni,:) = B;

    % update guess for inflation pi_t
    pi00 = pi_star;

end

% output
out = DPcoefss(1,:);

end
