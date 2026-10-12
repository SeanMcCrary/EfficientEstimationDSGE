function [logl,a,P,yhat,att] = kfilter(par,Y)

% Kalman Filter for the log linearized NKDMP model
% Input: data Y and structure par
%        Data is assumed to be in the following order: 
%        Y = [T.pi_obs, T.pi_obs2, T.n_obs, T.tg_obs, T.r_obs, T.s_obs, T.jf_obs]
%        Y = [T.pi_obs, T.pi_obs2, T.n_obs, T.tg_obs, T.r_obs, T.jf_obs]

    Tsim = length(Y);
try
    sol = nkdmp_loglinear_qz(par);

    % Order of the states: [n_{t-1}, z_t, d_t, m_t, s_t, a_t]
    % Order of policies: [n_t, pi_t, omega_t, q_t, tg_t]

    % State Transition Equation Matrices
    Tt = sol.M;

    ct = zeros(size(Tt,1),1);
    Qt = diag([par.sigz^2; par.sigd^2; par.sigm^2; par.sigs^2; par.siga^2]);
    Rt = [0 0 0 0 0;
          eye(5)];

    % Measurement Equation Matrices 
    Zt = [sol.P(2,:);
          sol.P(2,:);
          sol.P(1,:);
          sol.P(5,:);
          par.phipi*(sol.P(2,:)) + par.phiy*(sol.P(1,:)) + [0 par.phiy 0 1 0 0];
          0 0 0 0 1 0
          (1-par.alf)*sol.P(5,:) + [0 0 0 0 0 1]];
         %];

    dt = [log(par.PiSS); log(par.PiSS); 0; 0; log(par.Rss); 0; 0];
    %dt = [log(par.PiSS); log(par.PiSS); 0; 0; log(par.Rss); 0];

    % Part of the state space system that is not time-varying:
    Ht = diag([par.sigMEpi1; par.sigMEpi2; par.sigMEn; par.sigMEtg; par.sigMEr; par.sigMEs;  par.sigMEjf]).^2;
    %Ht = diag([par.sigMEpi1; par.sigMEpi2; par.sigMEn; par.sigMEtg; par.sigMEr;  par.sigMEjf]).^2;

    a0   = [0; 0; 0; 0; 0; 0];
    P0   = diag([std(Y(:,2)); ...
                 (par.sigz/(1-par.rz))^2; ...
                 (par.sigd/(1-par.rd))^2; ...
                 (par.sigm/(1-par.rm))^2; ...
                 (par.sigs/(1-par.rs))^2; ...
                 (par.siga/(1-par.ra))^2]);

    % initialization
    atp1   = a0;
    Ptp1   = P0;

    lafilter_y    = zeros(Tsim,size(Y,2));
    lafilter_logl = zeros(Tsim,1);
    lafilter_a(1,:)   = a0;
    lafilter_P(1,:,:) = P0;
    lafilter_at(1,:)  = a0;

    % loop
    for it=1:Tsim

        at = atp1;
        Pt = Ptp1;

        yt = Y(it,:)';

        % Kalman filter equations:
        vt   = yt - Zt*at - dt;
        Ft   = Zt*Pt*Zt' + Ht;
        att  = at + (Pt*Zt')/Ft*vt;
        atp1 = Tt*att + ct;
        Kt   = Tt*Pt*Zt'/Ft;
        Ptp1 = Tt*Pt*(Tt-Kt*Zt)' + Rt*Qt*Rt';
        lafilter_at(it,:)    = att;

        if it<Tsim
            lafilter_a(it+1,:)    = atp1;
            lafilter_P(it+1,:,:)  = Ptp1;
        end
        lafilter_y(it,:)  = Zt*att + dt; 
        lafilter_logl(it) = log(mvnpdf(vt,0,(Ft+Ft')/2)); % Remove roundoff asymmetry.
    end

    logl = sum(lafilter_logl);
    if logl == -inf
        logl = -1e6;
    end
    a    = lafilter_a;
    att  = lafilter_at;
    P    = lafilter_P;
    yhat = lafilter_y;
catch     
    logl = -inf;
     a    = [];
     att  = [];
     P    = [];
     yhat = [];
     return
end

%end
end
