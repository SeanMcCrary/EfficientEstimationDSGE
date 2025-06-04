function [logl,a,P,yhat] = tpfilter(par,Y)

% Filter using the Taylor Projection Approximations
% and the Time-varying KF
% Date: 4 feb 2025
% Authors: Eva Janssens & Sean McCrary
% Input: par (structure with all the parameters)
%        Y
% Output: logl is loglikelihood,
%         a filtered state (forecast),
%         P variance filtered state,
%         yhat is filtered observables

try
    warning('off','all')

    Tsim = length(Y);

    % System matrices that are not time-varying:
    Tt = diag([par.rhoz par.rhod par.rhom]);
    ct = zeros(3,1);
    Qt = diag([par.sigez^2 par.siged^2 par.sigem^2]);
    Rt = eye(3);

    Ht   = diag([par.sigMEy^2 par.sigMEpi^2 par.sigMEr^2]);
    sigz = par.sigez^2/(1-par.rhoz^2);
    sigd = par.siged^2/(1-par.rhod^2);
    sigr = par.sigem^2/(1-par.rhom^2);

    % initialization:
    a0   = zeros(3,1);
    P0   = diag([sigz sigd sigr]);

    atp1   = a0;
    Ptp1   = P0;

    lafilter_y    = zeros(Tsim,3);
    lafilter_logl = zeros(Tsim,1);

    lafilter_a(1,:)   = a0;
    lafilter_P(1,:,:) = P0;

    x0    = [par.piss; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0];
    x0    = nk_log_tp_coef(0,0,0,x0,par);

    % filter loop
    for it=1:Tsim
        
        % advance time
        at = atp1;
        Pt = Ptp1;

        yt = Y(it,:)'; % current observation

        % solution step
        nk_coef_t = nk_log_tp_coef(at(1), at(2), at(3), x0, par);

        a0 = nk_coef_t(1); a1 = nk_coef_t(2); 
        a2 = nk_coef_t(3); a3 = nk_coef_t(4);  % Inflation
        b0 = nk_coef_t(5); b1 = nk_coef_t(6); 
        b2 = nk_coef_t(7); b3 = nk_coef_t(8);  % Labor

        x0 = nk_coef_t;

        % built the system matrices based on current linear solution
        dt = [b0; (par.piss + a0); (par.Rss + par.phipi*a0 + par.phiy*b0)];
        
        Zt(1,:) = [(b1+1)  b2  b3];               % ouput 
        Zt(2,:) = [a1  a2  a3];                   % inflation 
        Zt(3,1) = par.phipi*a1 + par.phiy*(b1+1); % interest rate 
        Zt(3,2) = par.phipi*a2 + par.phiy*b2; 
        Zt(3,3) = par.phipi*a3 + par.phiy*b3 +1;
       
        % Kalman filter equations:
        vt   = yt - Zt*at - dt;
        Ft   = Zt*Pt*Zt' + Ht;
        att  = at + (Pt*Zt')/Ft*vt;
        atp1 = Tt*att + ct;
        Kt   = Tt*Pt*Zt'/Ft;
        Ptp1 = Tt*Pt*(Tt-Kt*Zt)' + Rt*Qt*Rt';
        if it<Tsim
            lafilter_a(it+1,:)    = atp1;
            lafilter_P(it+1,:,:)    = Ptp1;
        end
        lafilter_y(it,:)    = Zt*att + dt;
        lafilter_logl(it) = log(mvnpdf(vt,0,Ft));
    end

    logl = sum(lafilter_logl);
    if logl == -inf
        logl = -1e6;
    end
    a    = lafilter_a;
    P    = lafilter_P;
    yhat = lafilter_y;
catch
    logl = -inf;
    a    = [];
    P    = [];
    yhat = [];
    return
end

end