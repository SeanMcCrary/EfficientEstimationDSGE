function [logl,a,P,yhat,coefs] = tpfilter_TG(parvec,Y)

% Filter using the Taylor Projection Approximations
% and the Time-varying KF
% Date: 4 feb 2025
% Authors: Eva Janssens & Sean McCrary
% Input: parvec = [alfa,b,beta,c,delta,eta,rhoz,sige,sigtg,sigw]
%        Y
% Output: logl is loglikelihood,
%         a filtered state,
%         P variance filtered state,
%         yhat is filtered observables
%         coefs are the policy rule coefficients at each state
try
    warning('off','all')

    Tsim = length(Y);

    % Read out parameter vector:
    alfa  = parvec(1);
    b     = parvec(2);
    beta  = parvec(3);
    c     = parvec(4);
    delta = parvec(5);
    eta   = parvec(6);
    rhoz  = parvec(7);
    sige  = parvec(8);
    sigtg = parvec(9);

    [~,epsi_nodes,weight_nodes] = Monomials_2(1,sige^2);

    qss  = (1-beta*(1-delta))*c/((1-eta)*(1-b));     % ss vacancy-filling
    tgss = (qss^(-alfa) - 1)^(1/alfa);
    sigz = sige/sqrt(1-rhoz^2); % log tfp sd

    par.c     = c; 
    par.bet   = beta; 
    par.alfa  = alfa; 
    par.del   = delta;
    par.rhoz  = rhoz; 
    par.sige  = sige; 
    par.eta   = eta; 
    par.b     = b;
    par.sigz  = sigz; 
    par.egrid = epsi_nodes; 
    par.ewgt  = weight_nodes; 

    x0 = [tgss; 0];  % initial guess for coefficients

    % For the filter, we only have one state: alphat = log tfp;
    % Parts that are not time-varying:
    Tt = rhoz;
    ct = 0;
    Qt = sige^2;
    Rt = 1;

    Ht = sigtg^2;

    a0   = 0;
    P0   = sigz^2;

    % initialization
    atp1   = a0;
    Ptp1   = P0;

    lafilter_a(1,:)   = a0;
    lafilter_P(1,:,:) = P0;
    lafilter_y        = zeros(Tsim,1);
    lafilter_logl     = zeros(Tsim,1);
    coefs             = zeros(Tsim,2);
    % loop
    for it=1:Tsim

        at = atp1;
        Pt = Ptp1;
        
        yt = Y(it,:)';
        % Solution step:
        dmp_coef_t  = dmp_tpcoef_sym(at,x0,par);
        coefs(it,:) = dmp_coef_t;
        % Build system matrices
        x0 = dmp_coef_t;
        Zt = dmp_coef_t(2); 
        dt = dmp_coef_t(1); 
        % Kalman filter equations:
        vt   = yt - Zt*at - dt;
        Ft   = Zt*Pt*Zt' + Ht;
        att  = at + (Pt*Zt')/Ft*vt;
        atp1 = Tt*att + ct;
        Kt   = Tt*Pt*Zt'/Ft;
        Ptp1 = Tt*Pt*(Tt-Kt*Zt)' + Rt*Qt*Rt';
        if it<Tsim
            lafilter_a(it+1)    = atp1;
            lafilter_P(it+1)    = Ptp1;
        end
        lafilter_y(it,:)  = Zt*at + dt;
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