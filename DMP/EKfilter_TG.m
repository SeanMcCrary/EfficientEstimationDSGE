function [logl,a,yhat,P] = EKfilter_TG(parvec,Y)

% Filter using the Global Solution
% and the Extended Kalman Filter
% Date: 4 feb 2025
% Authors: Eva Janssens & Sean McCrary
% Input: parvec = [alfa,b,beta,c,delta,eta,rhoz,sige,sigtg,sigw]
%        Y
% Output: logl is loglikelihood, 
%         a filtered state, 
%         yhat is filtered observables

    rng(1);
    
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
    
    sigz = sige/sqrt(1-rhoz^2); % log tfp sd 
    
    [~,epsi_nodes,weight_nodes] = Monomials_2(1,sige^2);
    
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
    
    % Solve the model
    par.nsig    = 4;
    poly_degree = 9; % degree of polynomial approximation (5 is sufficient) 
    dmp_global_coefs = dmp_global_coef(par,poly_degree); 
    p = length(dmp_global_coefs)-1;
    
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
    ekffilter_a(1,:)   = a0;
    ekffilter_P(1,:,:) = P0;
    ekffilter_y    = zeros(Tsim,1);
    ekffilter_logl = zeros(Tsim,1);

    try
        % loop
        for it=1:Tsim
    
            at = atp1;
            Pt = Ptp1;
    
            yt = Y(it,:)';
    
            % Build the system matrices for the EKF at the current state:
            Zt_at = ((dmp_global_coefs')*((at.^(0:p))'))';
            dZt_dat = (((dmp_global_coefs(2:end)')*((at.^(0:(p-1)))'))');
            Zt = dZt_dat; 
            dt = Zt_at - dZt_dat*at;  
    
            % Kalman filter equations:
            vt   = yt - Zt*at - dt;
            Ft   = Zt*Pt*Zt' + Ht;
            att  = at + (Pt*Zt')/Ft*vt;
            atp1 = Tt*att + ct;
            Kt   = Tt*Pt*Zt'/Ft;
            Ptp1 = Tt*Pt*(Tt-Kt*Zt)' + Rt*Qt*Rt';
            if it<Tsim
                ekffilter_a(it+1)    = atp1;
                ekffilter_P(it+1)    = Ptp1;
            end
            ekffilter_y(it,:)    = Zt*at + dt;
            ekffilter_logl(it) = log(mvnpdf(vt,0,Ft));
        end
    
        logl = sum(ekffilter_logl);
        if logl == -inf
            logl = -1e6;
        end
        a    = ekffilter_a;
        P    = ekffilter_P;
        yhat = ekffilter_y;
    catch
        logl = -inf;
        a    = [];
        P    = [];
        yhat = [];
        return
    end

end