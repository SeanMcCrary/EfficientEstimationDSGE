function [logl,a,P,yhat] = kfilter_TG(parvec,Y)
% Filter using Log Linearization
% and the Kalman Filter
% Date: 4 feb 2025
% Authors: Eva Janssens & Sean McCrary
% Input: parvec = [alfa,b,beta,c,delta,eta,rhoz,sige,sigtg,sigw]
%        Y
% Output: logl is loglikelihood, 
%         a filtered state, 
%         P variance filtered state, 
%         yhat is filtered observables
try
    warning('off','all')

    Tsim = length(Y);
    % Read out parameter vector:
    alfa  = parvec(1);
    b     = parvec(2);
    bet   = parvec(3);
    eta   = parvec(6);
    rhoz  = parvec(7);
    sige  = parvec(8);
    sigtg = parvec(9);
    
    nss  = 0.945;                                 % ss employment 
    qss  = 0.7;                                   % ss vacancy-filling 
    tgss = 1;                                     % ss market tightness 

    ass  = qss/((1+tgss^alfa)^(-1/alfa));         % ss match efficiency 
    del  = qss*tgss*(1-nss)/((1-qss*tgss)*nss);   % ss separation rate 
    kap  = (1-eta)*(1-b)*qss/(1-bet*(1-del));     % vacancy cost
    c    = kap/ass; 
    
    % log-linear coefficient 
    gll  = ((qss^(1-alfa))*(tgss^(-alfa)))*((1-eta)/c)*((1-bet*(1-del)*rhoz)^(-1))/ass^(1-alfa);

    sigz = sige/sqrt(1-rhoz^2); % log tfp sd 
    
    % build all the system matrices
    Tt = rhoz;
    ct = 0;
    Qt = sige^2;
    Rt = 1;
    
    Zt = gll;
    dt = log(tgss);
    
    Ht   = sigtg^2; 
    
    a0   = 0;
    P0   = sigz^2;
    
    % initialization
    atp1   = a0;
    Ptp1   = P0;
    kfilter_a(1,:) = a0;
    kfilter_P(1,:,:) = P0;
    kfilter_y        = zeros(Tsim,1);
    kfilter_logl     = zeros(Tsim,1);
    
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
        if it<Tsim
            kfilter_a(it+1)    = atp1;
            kfilter_P(it+1)    = Ptp1;
        end
        kfilter_y(it,:)   = Zt*at + dt;
    
        try
            kfilter_logl(it) = log(mvnpdf(vt,0,Ft));
        catch
            logl = -1e6; a = []; P = []; yhat = [];
            return;
        end
    end
    
    kfilterlogl = sum(kfilter_logl);
    
    logl = sum(kfilterlogl);
    if logl == -inf
        logl = -1e6;
    end
    a    = kfilter_a;
    P    = kfilter_P;
    yhat = kfilter_y;
catch
    logl = -inf;
    a    = [];
    P    = [];
    yhat = [];
    return
end
end