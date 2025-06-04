function [logl,a,P,yhat,omgt,PolicyCoefs] = tpfilter_zlb(parvec,Y)
% Filter using the Taylor Projection Approximation and Time-Varying Kalman Filter
% Date: 11 Feb 2025
% Authors: Eva Janssens & Sean McCrary
%
% INPUTS:
%   parvec : parameter vector [31 x 1]
%   Y      : observed data [T x 5]
%
% OUTPUTS:
%   logl        : log-likelihood of observed data
%   a           : filtered state means
%   P           : filtered state covariances
%   yhat        : predicted observables
%   omgt        : flow surplus over time
%   PolicyCoefs : time-varying projection coefficients

try
    warning('off','all')  % suppress warnings (optional)

    %----------------------------------------------------------------------
    % Unpack parameters and steady-state transformations
    %----------------------------------------------------------------------

    Tsim      = length(Y);               % number of time periods
    rhoz      = parvec(1);               % TFP persistence
    sigez     = parvec(2);               % TFP shock std dev
    rhod      = parvec(3); siged = parvec(4);
    rhom      = parvec(5); sigem = parvec(6);
    rhos      = parvec(7); siges = parvec(8);
    rhoa      = parvec(9); sigea = parvec(10);
    alf       = parvec(16);              % matching elasticity
    sigMEpi   = parvec(26);              % measurement errors
    sigMEn    = parvec(27);
    sigMEtg   = parvec(28);
    sigMEr    = parvec(29);
    sigMEjf   = parvec(30);
    piss      = parvec(31);              % steady-state inflation
    rss       = parvec(24);              % steady-state nominal rate

    % Compute stationary shock std devs
    sigz = sigez / sqrt(1 - rhoz^2);
    sigd = siged / sqrt(1 - rhod^2);
    sigm = sigem / sqrt(1 - rhom^2);
    sigs = siges / sqrt(1 - rhos^2);
    siga = sigea / sqrt(1 - rhoa^2);

    %----------------------------------------------------------------------
    % Initial guess for policy coefficients
    %----------------------------------------------------------------------

    x0      = zeros(6,1);                        % state guess
    fun     = @(xx) Res_eval(x0, xx, parvec);    % residual function
    options = optimset('Display','off');         % silent fsolve
    x0      = fsolve(fun, zeros(28,1), options); % solve for policy coeffs
    x0      = x0(1:28);

    %----------------------------------------------------------------------
    % Constant system matrices
    %----------------------------------------------------------------------

    Qt = diag([sigez^2; siged^2; sigem^2; siges^2; sigea^2]);  % shock variances
    Rt = [zeros(1,5);
          eye(5)];                                             % shock loading on state
    Ht = diag([sigMEpi^2; sigMEn^2; sigMEtg^2; sigMEr^2; sigMEjf^2]); % measurement errors

    % Initial state and covariance
    a0  = zeros(6,1);
    P0  = diag([0.95 * std(Y(:,2))^2, sigz^2, sigd^2, sigm^2, sigs^2, siga^2]);

    atp1 = a0;
    Ptp1 = P0;

    %----------------------------------------------------------------------
    % Preallocation
    %----------------------------------------------------------------------

    lafilter_y        = zeros(Tsim, 5);        % predicted observables
    lafilter_logl     = zeros(Tsim, 1);        % period log-likelihoods
    lafilter_a(1,:)   = a0';                   % filtered states
    lafilter_P(1,:,:) = P0;                    % filtered variances
    omgt              = zeros(Tsim, 1);        % flow surplus
    PolicyCoefs       = zeros(Tsim, 35);       % time-varying policy coefficients

    %----------------------------------------------------------------------
    % Time-varying Kalman Filter loop
    %----------------------------------------------------------------------

    for it = 1:Tsim
        at = atp1;
        Pt = Ptp1;
        yt = Y(it,:)';

        % Solve for local projection coefficients at current state
        nkdmp_coef_t      = nkdmp_zlb_tpcoef(atp1, x0, parvec);
        PolicyCoefs(it,:) = nkdmp_coef_t;

        % Compute flow surplus
        omgt(it) = nkdmp_coef_t(15:21)' * [1; at];

        % Build time-varying system matrices
        auxR  = nkdmp_coef_t(30:end);
        auxJF = (1 - alf) * nkdmp_coef_t(2:7) + [0; 0; 0; 0; 0; 1];

        % Observation matrix Z_t and constant d_t
        Zt = [nkdmp_coef_t(9:14)';
              1 0 0 0 0 0;
              nkdmp_coef_t(2:7)';
              auxR';
              auxJF'];

        dt = [nkdmp_coef_t(8) + piss;
              0;
              nkdmp_coef_t(1);
              nkdmp_coef_t(29) + rss;
              (1 - alf) * nkdmp_coef_t(1)];

        % Transition matrix T_t and constant c_t
        Tt = [nkdmp_coef_t(23:28)';         % Nlag
              0 rhoz 0    0    0    0;      % z
              0 0    rhod 0    0    0;      % d
              0 0    0    rhom 0    0;      % m
              0 0    0    0    rhos 0;      % s
              0 0    0    0    0    rhoa];  % a

        ct = [nkdmp_coef_t(22); 0; 0; 0; 0; 0];

        %------------------------------------------------------------------
        % Kalman filter update step
        %------------------------------------------------------------------
        vt   = yt - Zt * at - dt;                          % prediction error
        Ft   = Zt * Pt * Zt' + Ht;                         % prediction variance
        att  = at + (Pt * Zt') / Ft * vt;                  % updated state
        atp1 = Tt * att + ct;                              % next period forecast
        Kt   = Tt * Pt * Zt' / Ft;                         % Kalman gain
        Ptp1 = Tt * Pt * (Tt - Kt * Zt)' + Rt * Qt * Rt';  % next period covariance

        % Store output
        if it < Tsim
            lafilter_a(it+1,:)   = atp1;
            lafilter_P(it+1,:,:) = Ptp1;
        end
        lafilter_y(it,:)  = Zt * att + dt;                % prediction
        lafilter_logl(it) = log(mvnpdf(vt, 0, Ft));       % log-likelihood
    end

    % Final outputs
    logl = sum(lafilter_logl);
    if logl == -inf
        logl = -1e6;  % Penalize underflow
    end
    a    = lafilter_a;
    P    = lafilter_P;
    yhat = lafilter_y;

catch
    % Return empty objects on failure
    logl = -inf;
    a    = [];
    P    = [];
    yhat = [];
    return
end

end
