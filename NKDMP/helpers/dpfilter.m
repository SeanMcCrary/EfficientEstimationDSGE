function [logl,a,P,yhat,att,Pfilt] = dpfilter(par,Y)

% Filter using the Dynamic Perturbation Approximations for NKDMP model
% without the ZLB.
%
% Inputs:
%   par : parameter structure
%   Y   : observed data ordered as
%         [pi_obs, pi_obs2, n_obs, tg_obs, r_obs, s_obs, jf_obs]
%
% Outputs:
%   logl   : log likelihood
%   a      : predicted state means, x_{t|t-1}
%   P      : predicted state covariances, P_{t|t-1}
%   yhat   : fitted observables
%   att    : filtered state means, x_{t|t}
%   Pfilt  : filtered state covariances, P_{t|t}

try
    warning('off','all')

    Tsim = length(Y);
    sol  = nkdmp_loglinear_qz(par);

    paramsDP = par;

    paramsDP.rho_z = par.rz;
    paramsDP.rho_d = par.rd;
    paramsDP.rho_m = par.rm;
    paramsDP.rho_s = par.rs;
    paramsDP.rho_a = par.ra;

    % Shock standard deviations
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

    paramsDP.theta0 = 0;
    paramsDP.pi0    = 0;
    paramsDP.omega0 = 0;
    paramsDP.n0     = 0;

    paramsDP.theta_n = sol.P(5,1);
    paramsDP.theta_z = sol.P(5,2);
    paramsDP.theta_d = sol.P(5,3);
    paramsDP.theta_m = sol.P(5,4);
    paramsDP.theta_s = sol.P(5,5);
    paramsDP.theta_a = sol.P(5,6);

    paramsDP.pi_n = sol.P(2,1);
    paramsDP.pi_z = sol.P(2,2);
    paramsDP.pi_d = sol.P(2,3);
    paramsDP.pi_m = sol.P(2,4);
    paramsDP.pi_s = sol.P(2,5);
    paramsDP.pi_a = sol.P(2,6);

    paramsDP.omega_n = sol.P(3,1);
    paramsDP.omega_z = sol.P(3,2);
    paramsDP.omega_d = sol.P(3,3);
    paramsDP.omega_m = sol.P(3,4);
    paramsDP.omega_s = sol.P(3,5);
    paramsDP.omega_a = sol.P(3,6);

    paramsDP.n_n = sol.P(1,1);
    paramsDP.n_z = sol.P(1,2);
    paramsDP.n_d = sol.P(1,3);
    paramsDP.n_m = sol.P(1,4);
    paramsDP.n_s = sol.P(1,5);
    paramsDP.n_a = sol.P(1,6);

    % Baseline auxiliary path for endogenous predetermined state
    paramsDP.nbar0 = 0;
    paramsDP.nbar1 = sol.P(1,1);
    paramsDP.nbar2 = sol.P(1,2);
    paramsDP.nbar3 = sol.P(1,3);
    paramsDP.nbar4 = sol.P(1,4);
    paramsDP.nbar5 = sol.P(1,5);
    paramsDP.nbar6 = sol.P(1,6);

    Naux = 10;

    % System matrices that are not time-varying
    Qt = diag([par.sigz^2; ...
               par.sigd^2; ...
               par.sigm^2; ...
               par.sigs^2; ...
               par.siga^2]);

    Rt = [0 0 0 0 0;
          eye(5)];

    Ht = diag([par.sigMEpi1; ...
               par.sigMEpi2; ...
               par.sigMEn; ...
               par.sigMEtg; ...
               par.sigMEr; ...
               par.sigMEs; ...
               par.sigMEjf]).^2;

    % Initial state mean and covariance
    a0 = [0; 0; 0; 0; 0; 0];

    P0 = diag([std(Y(:,2)); ...
               (par.sigz/(1-par.rz))^2; ...
               (par.sigd/(1-par.rd))^2; ...
               (par.sigm/(1-par.rm))^2; ...
               (par.sigs/(1-par.rs))^2; ...
               (par.siga/(1-par.ra))^2]);

    atp1 = a0;
    Ptp1 = P0;

    nStates = length(a0);

    % Storage
    lafilter_y     = zeros(Tsim,size(Y,2));
    lafilter_logl  = zeros(Tsim,1);
    lafilter_a     = zeros(Tsim,nStates);
    lafilter_att   = zeros(Tsim,nStates);
    lafilter_P     = zeros(Tsim,nStates,nStates);
    lafilter_Pfilt = zeros(Tsim,nStates,nStates);

    % Filtering loop
    for it = 1:Tsim

        % One-step-ahead state distribution
        at = atp1;
        Pt = Ptp1;

        lafilter_a(it,:)   = at';
        lafilter_P(it,:,:) = Pt;

        yt = Y(it,:)';

        sol_dp = nkdmp_nozlb_dp_coef(atp1,paramsDP,Naux);

        % State order:
        % [n_{t-1}, z_t, d_t, m_t, s_t, a_t]
        %
        % Policy order:
        % [n_t, pi_t, omega_t, q_t, theta_t]

        Tt = [sol_dp.b0(23:28);
              0 par.rz 0      0      0      0;
              0 0      par.rd 0      0      0;
              0 0      0      par.rm 0      0;
              0 0      0      0      par.rs 0;
              0 0      0      0      0      par.ra];

        ct = [sol_dp.b0(22) - sol_dp.b0(23:28)*atp1;
              0;
              0;
              0;
              0;
              0];

        % Observed data order:
        % [pi_obs, pi_obs2, n_obs, tg_obs, r_obs, s_obs, jf_obs]

        auxR = par.phipi*sol_dp.b0(9:14) ...
             + par.phiy*sol_dp.b0(23:28) ...
             + [0 par.phiy 0 1 0 0];

        auxRint = log(par.Rss) ...
                + par.phipi*sol_dp.b0(8) ...
                - par.phipi*sol_dp.b0(9:14)*atp1 ...
                + par.phiy*sol_dp.b0(22) ...
                - par.phiy*sol_dp.b0(23:28)*atp1;

        Zt = [sol_dp.b0(9:14);
              sol_dp.b0(9:14);
              sol_dp.b0(23:28);
              sol_dp.b0(2:7);
              auxR;
              0 0 0 0 1 0;
              (1-par.alf)*sol_dp.b0(2:7) + [0 0 0 0 0 1]];

        dt = [sol_dp.b0(8) ...
              - sol_dp.b0(9:14)*atp1 ...
              + log(par.PiSS);

              sol_dp.b0(8) ...
              - sol_dp.b0(9:14)*atp1 ...
              + log(par.PiSS);

              sol_dp.b0(22) ...
              - sol_dp.b0(23:28)*atp1;

              sol_dp.b0(1) ...
              - sol_dp.b0(2:7)*atp1;

              auxRint;

              0;

              (1-par.alf)*sol_dp.b0(1) ...
              - (1-par.alf)*sol_dp.b0(2:7)*atp1];

        % Innovation and innovation covariance
        vt = yt - Zt*at - dt;
        Ft = Zt*Pt*Zt' + Ht;

        % Kalman filtering gain
        Gt = (Pt*Zt')/Ft;

        % Filtered state mean: x_{t|t}
        att = at + Gt*vt;

        % Filtered covariance: P_{t|t}
        % Joseph form improves numerical stability.
        Istate = eye(nStates);

        Ptt = (Istate-Gt*Zt)*Pt*(Istate-Gt*Zt)' ...
            + Gt*Ht*Gt';

        % Remove small numerical asymmetries
        Ptt = (Ptt+Ptt')/2;

        % Store filtered quantities
        lafilter_att(it,:)     = att';
        lafilter_Pfilt(it,:,:) = Ptt;

        % One-step-ahead state mean
        atp1 = Tt*att + ct;

        % Keep the original covariance recursion
        Kt   = Tt*Pt*Zt'/Ft;
        Ptp1 = Tt*Pt*(Tt-Kt*Zt)' + Rt*Qt*Rt';

        % Remove small numerical asymmetries
        Ptp1 = (Ptp1+Ptp1')/2;

        % Fitted observations and log likelihood
        lafilter_y(it,:)  = (Zt*att + dt)';
        lafilter_logl(it) = log(mvnpdf(vt,0,Ft));
    end

    logl = sum(lafilter_logl);

    if logl == -inf
        logl = -1e6;
    end

    a     = lafilter_a;
    P     = lafilter_P;
    yhat  = lafilter_y;
    att   = lafilter_att;
    Pfilt = lafilter_Pfilt;

catch
    logl   = -inf;
    a      = [];
    P      = [];
    yhat   = [];
    att    = [];
    Pfilt  = [];
    return
end

end