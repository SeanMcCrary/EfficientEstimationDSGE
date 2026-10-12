function [logl,a,P,yhat,att,Pfilt] = dp_zlb_filter(par,Y,bindZLB)
% -------------------------------------------------------------------------
% dp_zlb_filter
%
% Time-varying Gaussian filter for the NK-DMP model with an occasionally
% binding ZLB, using branch-specific Dynamic Perturbation (DP) local
% approximations.
%
% The observed regime b_t is treated as known without error. At each date:
%   (i)   the observed regime selects the branch-specific transition and
%         measurement equations;
%   (ii)  the continuous-data likelihood is evaluated under that branch;
%   (iii) an additional regime-likelihood term is added, equal to the
%         predictive probability of the observed regime under the
%         unconstrained shadow-rate rule.
%
% This is a Gaussian approximation to the exact observed-regime likelihood.
%
% INPUTS
%   par      : parameter structure
%   Y        : T x 6 data matrix, ordered as
%                [pi_obs, pi_obs2, n_obs, tg_obs, r_obs, s_obs]
%   bindZLB  : T x 1 observed regime indicator
%                1 = binding regime
%                0 = slack regime
%
% OUTPUTS
%   logl     : total approximate log-likelihood
%   a        : (T+1) x 6 predicted state means, a_{t|t-1}
%   P        : (T+1) x 6 x 6 predicted state covariances
%   yhat     : T x 6 fitted observables, E[y_t | Y_{1:t}]
%   att      : T x 6 filtered state means, a_{t|t}
%   Pfilt    : T x 6 x 6 filtered state covariances, P_{t|t}
%
% STATE ORDER
%   x_t = [n_{t-1}, z_t, d_t, m_t, s_t, a_t]'
%
% POLICY BLOCK ORDER
%   b0 = [theta0, theta_slopes(6), ...
%         pi0,    pi_slopes(6), ...
%         omega0, omega_slopes(6), ...
%         n0,     n_slopes(6)]
%
% NOTES
%   - The filter solves the DP object at the current predicted state.
%   - The observed regime determines which current branch is used.
%   - The regime probability term is computed from the unconstrained
%     shadow-rate equation, which is affine under the local approximation.
%   - If your observed rate is measured relative to a nonzero lower bound,
%     set par.rLB accordingly; otherwise the default lower bound is zero.
% -------------------------------------------------------------------------

    %% --------------------------------------------------------------------
    % 0. Basic checks

    if nargin < 3
        error('dp_zlb_filter requires par, Y, and bindZLB.');
    end

    if size(Y,2) ~= 7
        error('Y must be T x 6 with columns [pi_obs, pi_obs2, n_obs, tg_obs, r_obs, s_obs].');
    end

    Tsim = size(Y,1);

    bindZLB = bindZLB(:);
    if numel(bindZLB) ~= Tsim
        error('bindZLB must be T x 1.');
    end

    % Lower bound for the observed rate variable
    if isfield(par,'rLB')
        rLB = par.rLB;
    else
        rLB = 0;
    end

    %% --------------------------------------------------------------------
    % 1. Log-linear solution and DP parameter structs

    sol = nkdmp_loglinear_qz(par);

    paramsDP = struct();

    % AR coefficients
    paramsDP.rho_z = par.rz;
    paramsDP.rho_d = par.rd;
    paramsDP.rho_m = par.rm;
    paramsDP.rho_s = par.rs;
    paramsDP.rho_a = par.ra;

    % Shock std devs
    paramsDP.sig_z = par.sigz;
    paramsDP.sig_d = par.sigd;
    paramsDP.sig_m = par.sigm;
    paramsDP.sig_s = par.sigs;
    paramsDP.sig_a = par.siga;

    % Structural parameters
    paramsDP.tau    = par.tau;
    paramsDP.phi_pi = par.phipi;
    paramsDP.phi_y  = par.phiy;
    paramsDP.beta   = par.bet;
    paramsDP.alpha  = par.alf;

    paramsDP.Q      = par.Q;
    paramsDP.gamma  = par.gam;
    paramsDP.delta  = par.del;
    paramsDP.N      = par.N;
    paramsDP.nu     = par.nu;
    paramsDP.kappa  = par.kap;
    paramsDP.chi    = par.chi;

    % Terminal linear rule from log-linear solution
    paramsDP.theta0 = 0;
    paramsDP.pi0    = 0;
    paramsDP.omega0 = 0;
    paramsDP.n0     = 0;

    paramsDP.theta_n = sol.P(5,1);  paramsDP.theta_z = sol.P(5,2);  paramsDP.theta_d = sol.P(5,3);
    paramsDP.theta_m = sol.P(5,4);  paramsDP.theta_s = sol.P(5,5);  paramsDP.theta_a = sol.P(5,6);

    paramsDP.pi_n    = sol.P(2,1);  paramsDP.pi_z    = sol.P(2,2);  paramsDP.pi_d    = sol.P(2,3);
    paramsDP.pi_m    = sol.P(2,4);  paramsDP.pi_s    = sol.P(2,5);  paramsDP.pi_a    = sol.P(2,6);

    paramsDP.omega_n = sol.P(3,1);  paramsDP.omega_z = sol.P(3,2);  paramsDP.omega_d = sol.P(3,3);
    paramsDP.omega_m = sol.P(3,4);  paramsDP.omega_s = sol.P(3,5);  paramsDP.omega_a = sol.P(3,6);

    paramsDP.n_n     = sol.P(1,1);  paramsDP.n_z     = sol.P(1,2);  paramsDP.n_d     = sol.P(1,3);
    paramsDP.n_m     = sol.P(1,4);  paramsDP.n_s     = sol.P(1,5);  paramsDP.n_a     = sol.P(1,6);

    % Baseline auxiliary path for endogenous predetermined state
    paramsDP.nbar0 = 0;
    paramsDP.nbar1 = sol.P(1,1);
    paramsDP.nbar2 = sol.P(1,2);
    paramsDP.nbar3 = sol.P(1,3);
    paramsDP.nbar4 = sol.P(1,4);
    paramsDP.nbar5 = sol.P(1,5);
    paramsDP.nbar6 = sol.P(1,6);

    % ZLB struct: same terminal rule on both branches
    paramsDPzlb = paramsDP;

    if isfield(par,'rbar')
        paramsDPzlb.rbar = par.rbar;
    elseif isfield(par,'Rss')
        paramsDPzlb.rbar = log(par.Rss);
    else
        error('Need par.rbar or par.Rss to define the ZLB threshold.');
    end

    paramsDPzlb.theta0_U = 0; paramsDPzlb.pi0_U = 0; paramsDPzlb.omega0_U = 0; paramsDPzlb.n0_U = 0;
    paramsDPzlb.theta0_C = 0; paramsDPzlb.pi0_C = 0; paramsDPzlb.omega0_C = 0; paramsDPzlb.n0_C = 0;

    paramsDPzlb.theta_n_U = paramsDP.theta_n;  paramsDPzlb.theta_z_U = paramsDP.theta_z;
    paramsDPzlb.theta_d_U = paramsDP.theta_d;  paramsDPzlb.theta_m_U = paramsDP.theta_m;
    paramsDPzlb.theta_s_U = paramsDP.theta_s;  paramsDPzlb.theta_a_U = paramsDP.theta_a;

    paramsDPzlb.pi_n_U    = paramsDP.pi_n;     paramsDPzlb.pi_z_U    = paramsDP.pi_z;
    paramsDPzlb.pi_d_U    = paramsDP.pi_d;     paramsDPzlb.pi_m_U    = paramsDP.pi_m;
    paramsDPzlb.pi_s_U    = paramsDP.pi_s;     paramsDPzlb.pi_a_U    = paramsDP.pi_a;

    paramsDPzlb.omega_n_U = paramsDP.omega_n;  paramsDPzlb.omega_z_U = paramsDP.omega_z;
    paramsDPzlb.omega_d_U = paramsDP.omega_d;  paramsDPzlb.omega_m_U = paramsDP.omega_m;
    paramsDPzlb.omega_s_U = paramsDP.omega_s;  paramsDPzlb.omega_a_U = paramsDP.omega_a;

    paramsDPzlb.n_n_U     = paramsDP.n_n;      paramsDPzlb.n_z_U     = paramsDP.n_z;
    paramsDPzlb.n_d_U     = paramsDP.n_d;      paramsDPzlb.n_m_U     = paramsDP.n_m;
    paramsDPzlb.n_s_U     = paramsDP.n_s;      paramsDPzlb.n_a_U     = paramsDP.n_a;

    paramsDPzlb.theta_n_C = paramsDP.theta_n;  paramsDPzlb.theta_z_C = paramsDP.theta_z;
    paramsDPzlb.theta_d_C = paramsDP.theta_d;  paramsDPzlb.theta_m_C = paramsDP.theta_m;
    paramsDPzlb.theta_s_C = paramsDP.theta_s;  paramsDPzlb.theta_a_C = paramsDP.theta_a;

    paramsDPzlb.pi_n_C    = paramsDP.pi_n;     paramsDPzlb.pi_z_C    = paramsDP.pi_z;
    paramsDPzlb.pi_d_C    = paramsDP.pi_d;     paramsDPzlb.pi_m_C    = paramsDP.pi_m;
    paramsDPzlb.pi_s_C    = paramsDP.pi_s;     paramsDPzlb.pi_a_C    = paramsDP.pi_a;

    paramsDPzlb.omega_n_C = paramsDP.omega_n;  paramsDPzlb.omega_z_C = paramsDP.omega_z;
    paramsDPzlb.omega_d_C = paramsDP.omega_d;  paramsDPzlb.omega_m_C = paramsDP.omega_m;
    paramsDPzlb.omega_s_C = paramsDP.omega_s;  paramsDPzlb.omega_a_C = paramsDP.omega_a;

    paramsDPzlb.n_n_C     = paramsDP.n_n;      paramsDPzlb.n_z_C     = paramsDP.n_z;
    paramsDPzlb.n_d_C     = paramsDP.n_d;      paramsDPzlb.n_m_C     = paramsDP.n_m;
    paramsDPzlb.n_s_C     = paramsDP.n_s;      paramsDPzlb.n_a_C     = paramsDP.n_a;

    %% --------------------------------------------------------------------
    % 2. Fixed system matrices and filter initialization

    Naux = 5;

    % State innovation covariance for exogenous states only
    Qt = diag([par.sigz^2; par.sigd^2; par.sigm^2; par.sigs^2; par.siga^2]);

    % Maps the 5 exogenous shocks into the 6-state system
    Rt = [zeros(1,5); eye(5)];

    % Measurement error covariance
    Ht = diag([par.sigMEpi1; par.sigMEpi2; par.sigMEn; par.sigMEtg; par.sigMEr; par.sigMEs; par.sigMEjf]).^2;

    % Initial predicted state
    a0 = zeros(6,1);

    % Initial covariance
    y_n = Y(:,3);
    y_n = y_n(isfinite(y_n));
    if isempty(y_n)
        var_n0 = 1e-4;
    else
        var_n0 = max(var(y_n), 1e-6);
    end

    P0 = diag([ ...
        var_n0; ...
        par.sigz^2 / max(1 - par.rz^2, 1e-8); ...
        par.sigd^2 / max(1 - par.rd^2, 1e-8); ...
        par.sigm^2 / max(1 - par.rm^2, 1e-8); ...
        par.sigs^2 / max(1 - par.rs^2, 1e-8); ...
        par.siga^2 / max(1 - par.ra^2, 1e-8) ]);

    atp1 = a0;
    Ptp1 = P0;

    % Storage
    a    = zeros(Tsim+1,6);
    P    = zeros(Tsim+1,6,6);
    att  = zeros(Tsim,6);
    Pfilt = zeros(Tsim,6,6);
    yhat = zeros(Tsim,7);
    ll_t = zeros(Tsim,1);

    a(1,:)   = a0.';
    P(1,:,:) = P0;

    %% --------------------------------------------------------------------
    % 3. Filter loop

    for t = 1:Tsim

        % Current observation
        yt = Y(t,:).';

        % Current predicted state
        at = atp1;
        Pt = Ptp1;

        % Solve DP approximation at the current predicted state
        sol_dp = nkdmp_zlb_dp_coef(at, paramsDPzlb, Naux);

        % -----------------------------------------------------------------
        % 3a. Select branch using the OBSERVED regime

        if bindZLB(t) == 0
            bcur = real(sol_dp.b0U(:));
            useBind = 0;
        else
            bcur = real(sol_dp.b0C(:));
            useBind = 1;
        end

        % Unconstrained branch coefficients, used to compute the predictive
        % probability of the observed regime
        bU = real(sol_dp.b0U(:));

        % -----------------------------------------------------------------
        % 3b. Transition equation under the observed regime:
        %     x_{t+1} = T_t x_t + c_t + R eta_{t+1}

        Tt = [bcur(23:28).'; ...
              0 par.rz 0      0      0      0; ...
              0 0      par.rd 0      0      0; ...
              0 0      0      par.rm 0      0; ...
              0 0      0      0      par.rs 0; ...
              0 0      0      0      0      par.ra];

        ct = [bcur(22) - bcur(23:28)'*at; 0; 0; 0; 0; 0];

        % -----------------------------------------------------------------
        % 3c. Measurement equation under the observed regime:
        %     y_t = Z_t x_t + d_t + eps_t
        %
        % Observable order:
        %   1. pi_obs
        %   2. pi_obs2
        %   3. n_obs
        %   4. tg_obs
        %   5. r_obs
        %   6. s_obs

        if useBind == 0
            % Current nominal rate under the slack branch
            auxR    = par.phipi*bcur(9:14).' + par.phiy*bcur(23:28).' + [0 par.phiy 0 1 0 0];
            auxRint = log(par.Rss) ...
                    + par.phipi*(bcur(8)  - bcur(9:14)'*at) ...
                    + par.phiy *(bcur(22) - bcur(23:28)'*at);
        else
            % Current nominal rate fixed at the lower bound in the binding branch
            auxR    = zeros(1,6);
            auxRint = rLB;
        end

        Zt = [bcur(9:14).'; ...
              bcur(9:14).'; ...
              bcur(23:28).'; ...
              bcur(2:7).'; ...
              auxR; ...
              0 0 0 0 1 0; ...
              (1-par.alf)*bcur(2:7)' + [0 0 0 0 0 1]];

        dt = [bcur(8)  - bcur(9:14)'*at + log(par.PiSS); ...
              bcur(8)  - bcur(9:14)'*at + log(par.PiSS); ...
              bcur(22) - bcur(23:28)'*at; ...
              bcur(1)  - bcur(2:7)'*at; ...
              auxRint; ...
              0; ...
              (1-par.alf)*bcur(1) - (1-par.alf)*bcur(2:7)'*at];

        % -----------------------------------------------------------------
        % 3d. Continuous-data Gaussian likelihood under the observed regime

        vt = yt - Zt*at - dt;
        Ft = Zt*Pt*Zt.' + Ht;
        Ft = 0.5*(Ft + Ft.');   % enforce symmetry

        [L,pchol] = chol(Ft,'lower');
        if pchol ~= 0 || any(~isfinite(Ft(:)))
            logl = -1e10;
            a    = [];
            att  = [];
            P    = [];
            yhat = [];
            Pfilt = [];
            return
        end

        innovStd = L \ vt;
        logdetF  = 2*sum(log(diag(L)));
        nObs     = size(Y,2);

        ll_y = -0.5 * (nObs*log(2*pi) + logdetF + innovStd.'*innovStd);

        % -----------------------------------------------------------------
        % 3e. Regime-likelihood term from the unconstrained shadow rate
        %
        % Under the local approximation, the unconstrained current rate is
        % affine in x_t:
        %   r_t^U(x_t) = beta_r' x_t + alpha_r.
        %
        % Given x_t | Y_{1:t-1} ~ N(at,Pt), the predictive shadow rate is
        % Gaussian, so the predictive binding probability is available in
        % closed form.

        beta_r_U  = par.phipi*bU(9:14).' + par.phiy*bU(23:28).' + [0 par.phiy 0 1 0 0];
        alpha_r_U = log(par.Rss) ...
                  + par.phipi*(bU(8)  - bU(9:14)'*at) ...
                  + par.phiy *(bU(22) - bU(23:28)'*at);

        mu_r  = beta_r_U*at + alpha_r_U;
        var_r = beta_r_U*Pt*beta_r_U.';
        sig_r = sqrt(max(var_r,1e-12));

        % Predictive probability that the unconstrained shadow rate falls
        % below the lower bound
        pBind = 0.5 * erfc((mu_r - rLB) / (sqrt(2)*sig_r));
        pBind = min(max(pBind,1e-12), 1-1e-12);

        if bindZLB(t) == 1
            ll_b = log(pBind);
        else
            ll_b = log(1 - pBind);
        end

        % Total one-step approximate log likelihood
        ll_t(t) = ll_y + ll_b;

        % -----------------------------------------------------------------
        % 3f. Kalman update under the observed regime branch

        Kt = (Pt*Zt.') / Ft;
        att_t = at + Kt*vt;

        I6 = eye(6);
        Pt_upd = (I6 - Kt*Zt)*Pt*(I6 - Kt*Zt).' + Kt*Ht*Kt.';
        Pt_upd = 0.5*(Pt_upd + Pt_upd.');

        % -----------------------------------------------------------------
        % 3g. One-step prediction under the observed regime branch

        atp1 = Tt*att_t + ct;
        Ptp1 = Tt*Pt_upd*Tt.' + Rt*Qt*Rt.';
        Ptp1 = 0.5*(Ptp1 + Ptp1.');

        % -----------------------------------------------------------------
        % 3h. Store outputs

        att(t,:)    = att_t.';
        Pfilt(t,:,:) = Pt_upd;
        yhat(t,:)   = (Zt*att_t + dt).';
        a(t+1,:)    = atp1.';
        P(t+1,:,:)  = Ptp1;
    end

    %% --------------------------------------------------------------------
    % 4. Total log-likelihood

    logl = sum(ll_t);

    if ~isfinite(logl)
        logl = -1e10;
    end

end
