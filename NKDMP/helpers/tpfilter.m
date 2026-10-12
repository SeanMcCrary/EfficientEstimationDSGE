function [logl,a,P,yhat,att] = tpfilter(par,Y)

% Filter using the Taylor Projection Approximations for NKDMP model (no ZLB)
% Input: data Y and structure par
%        Data is assumed to be in the following order: 
%        Y = [T.pi_obs, T.pi_obs2, T.n_obs, T.tg_obs, T.r_obs, T.s_obs, T.jf_obs]

try
    warning('off','all')

    Tsim  = length(Y);
    sol = nkdmp_loglinear_qz(par);
  
    parvec = [par.rz;par.sigz;par.rd;par.sigd;par.rm;par.sigm;par.rs;par.sigs; ...
              par.ra;par.siga;par.tau;par.phipi;par.phiy;par.kap;par.bet;par.alf;...
              par.Q;par.chi;par.gam;par.del;par.N;par.nu]; 

    B0 = [0;sol.P(5,:)';0;sol.P(2,:)';0;sol.P(3,:)';0;sol.P(1,:)'];

    % System matrices that are not time-varying
    Qt = diag([par.sigz^2; par.sigd^2; par.sigm^2; par.sigs^2; par.siga^2]);
    Rt = [0 0 0 0 0;
          eye(5)];
    Ht = diag([par.sigMEpi1; par.sigMEpi2; par.sigMEn; par.sigMEtg; par.sigMEr; par.sigMEs; par.sigMEjf]).^2;
    % Ht = diag([par.sigMEpi1; par.sigMEpi2; par.sigMEn; par.sigMEtg; par.sigMEr; par.sigMEjf]).^2;

    % initialization
    a0   = [0; 0; 0; 0; 0; 0];
    P0   = diag([std(Y(:,2)); ...
                 (par.sigz/(1-par.rz))^2; ...
                 (par.sigd/(1-par.rd))^2; ...
                 (par.sigm/(1-par.rm))^2; ...
                 (par.sigs/(1-par.rs))^2; ...
                 (par.siga/(1-par.ra))^2]);

    atp1              = a0;
    Ptp1              = P0;

    lafilter_y    = zeros(Tsim,size(Y,2));
    lafilter_logl = zeros(Tsim,1);
    lafilter_a(1,:)   = a0;
    lafilter_P(1,:,:) = P0;
    lafilter_att(1,:) = a0;

    % loop
    for it=1:Tsim

        at = atp1;
        Pt = Ptp1;

        yt = Y(it,:)';

        nkdmp_coef_t = nkdmp_tpcoef(atp1,B0,parvec);
       
        % Order of the states: [n_{t-1}, z_t, d_t, m_t, s_t, a_t]
        % Order of policies: [n_t, pi_t, omega_t, q_t, tg_t]

        Tt = [nkdmp_coef_t.B(23:28)';              % Nlag
              0 par.rz 0     0     0     0;        % Z
              0  0   par.rd  0     0     0;        % D
              0  0   0     par.rm  0     0;        % M
              0  0   0     0     par.rs  0;        % S
              0  0   0     0     0     par.ra];    % A      

        ct = [nkdmp_coef_t.B(22) -  nkdmp_coef_t.B(23:28)'*atp1; 0; 0; 0; 0; 0];

        % Order of data:   Y = [T.pi_obs, T.pi_obs2, T.n_obs, T.tg_obs, T.r_obs, T.s_obs, T.jf_obs]

        auxR = par.phipi*nkdmp_coef_t.B(9:14)' + par.phiy*nkdmp_coef_t.B(23:28)' + [0 par.phiy 0 1 0 0];
        auxRint = log(par.Rss) + par.phipi*nkdmp_coef_t.B(8) - par.phipi*nkdmp_coef_t.B(9:14)'*atp1 + par.phiy*nkdmp_coef_t.B(22) - par.phiy*nkdmp_coef_t.B(23:28)'*atp1;

        Zt = [nkdmp_coef_t.B(9:14)' ;
              nkdmp_coef_t.B(9:14)' ; 
              nkdmp_coef_t.B(23:28)';
              nkdmp_coef_t.B(2:7)'  ;
              auxR                  ;
              0  0  0  0  1  0      ;
              (1-par.alf)*nkdmp_coef_t.B(2:7)' + [0 0 0 0 0 1]]; 
        
        dt = [nkdmp_coef_t.B(8) - nkdmp_coef_t.B(9:14)'*atp1 + log(par.PiSS); 
              nkdmp_coef_t.B(8) - nkdmp_coef_t.B(9:14)'*atp1 + log(par.PiSS); 
              nkdmp_coef_t.B(22) -  nkdmp_coef_t.B(23:28)'*atp1; 
              nkdmp_coef_t.B(1) - nkdmp_coef_t.B(2:7)'*atp1; 
              auxRint; 
              0;                                            
              (1-par.alf)*nkdmp_coef_t.B(1) - (1-par.alf)*nkdmp_coef_t.B(2:7)'*atp1];

        % Kalman filter equations:
        vt   = yt - Zt*at - dt;
        Ft   = Zt*Pt*Zt' + Ht;
        att  = at + (Pt*Zt')/Ft*vt;
        atp1 = Tt*att + ct;
        Kt   = Tt*Pt*Zt'/Ft;
        Ptp1 = Tt*Pt*(Tt-Kt*Zt)' + Rt*Qt*Rt';
        lafilter_att(it,:) = att;
        if it<Tsim
            lafilter_a(it+1,:)    = atp1;
            lafilter_P(it+1,:,:)  = Ptp1;
        end
        lafilter_y(it,:)  = Zt*att + dt; % sean changed at--> att 
        lafilter_logl(it) = log(mvnpdf(vt,0,Ft));
    end

    logl = sum(lafilter_logl);
    if logl == -inf
        logl = -1e6;
    end
    a    = lafilter_a;
    att  = lafilter_att;
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

end