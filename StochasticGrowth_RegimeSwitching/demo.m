clear;  clc; 
rng(52);

% Set parameters for Stochastic Volatility with Regime switching
p0  = 0.95;  % probability to stay in state 0   
rz0 = 0.70;  % productivity persistence in state 0
sz0 = 0.01;  % stdev. of productivity shocks in state 0 
p1  = 0.95;  % probability to stay in state 1
rz1 = 0.975; % productivity persistence in state 1
sz1 = 0.002; % stdev. of productivity shocks in state 1
PI  = [p0 (1-p0); 
       (1-p1) p1]; 
mc  = dtmc(PI); 

tau = 2;     % Risk aversion
bet = 1.04^(-1/4);    
alf = 0.36; 
del = 0.025; % Depreciation

% Collect all parameters in a vector
pvec = [rz0;sz0;rz1;sz1;p0;p1;tau;bet;alf;del];

m    = 2;                % number of Markov states
rhoz = [rz0 rz1];
sigz = [sz0 sz1];

% Solve model at the steady state: 
B0   = [0;0.4;0.2;0;0.95;0.05;0;0.4;0.2;0;0.95;0.05]; % (initial guess)
xt   = [0.0;0.0]; 
Bss  = rbcar1mc_tpcoef(xt,B0,pvec); 

% Simulate a time series 
T    = 500; 
burn = 250; 
st   = simulate(mc,T)-1;   % simulate from the Markov chain
zt   = zeros(T,1); 
et   = normrnd(0,1,T,1); 
% simulate productivity sequence
for ti=2:T
    zt(ti) = (rz0+st(ti)*(rz1-rz0))*zt(ti-1) + (sz0+st(ti)*(sz1-sz0))*et(ti);
end 

kt = zeros(T,1); 
Bt = zeros(T,12); 
ct = zeros(T,1); 
yt = zeros(T,1); 

for ti=2:T

    % solution step for simulation:
    xt     = [kt(ti-1);zt(ti)];  % current state
    Btemp  = rbcar1mc_tpcoef(xt,B0,pvec);  % current local solution
    Bt(ti,:) = [Btemp(1,:)';Btemp(2,:)'];  % collect terms
    
    % observables: follow from the policy rules
    kt(ti) = (1-st(ti))*(Btemp(1,4)+Btemp(1,5)*xt(1)+Btemp(1,6)*xt(2)) + ...
                 st(ti)*(Btemp(2,4)+Btemp(2,5)*xt(1)+Btemp(2,6)*xt(2));
    ct(ti) = (1-st(ti))*(Btemp(1,1)+Btemp(1,2)*xt(1)+Btemp(1,3)*xt(2)) + ...
                 st(ti)*(Btemp(2,1)+Btemp(2,2)*xt(1)+Btemp(2,3)*xt(2));
    yt(ti) = zt(ti) + alf*kt(ti-1); 

end 
 

% observables (remove burn-in)
cts = ct(burn+1:end); 
yt  = yt(burn+1:end); 
zt  = zt(burn+1:end); 
st  = st(burn+2:end); 
kt  = kt(burn:end-1); 

B0   = [Bss(1,:)';Bss(2,:)']; 

% collect the data observables
simy = [yt cts]';
Tsim = length(simy);

% add measurement error
sigmey = 0.04*std(simy(1,:));
sigmec = 0.04*std(simy(2,:));
H      = diag([sigmey^2 sigmec^2]);
simy   = simy + sqrt(H)*randn(2,Tsim);
simy   = simy';

%% Filter

coefs = zeros(T,size(Bss,1),size(Bss,2)); 

% Initalization of the filter:
atp1{1}   = [0; 0]; 
at{1}     = atp1{1};
att{1}    = atp1;
Ptp1{1}   = ([std(simy(1,:))^2 0; 0 sigz(1)^2/(1-rhoz(1)^2)]);
Pt{1}     = Ptp1{1};
atp1{2}   = [0; 0]; 
att{2}    = atp1;
Ptp1{2}   = ([std(simy(1,:))^2 0; 0 sigz(2)^2/(1-rhoz(2)^2)]);
at{2}     = atp1{2};
Pt{2}     = Ptp1{2};
lafilter_y{1}  = zeros(Tsim,2);
lafilter_y{2}  = zeros(Tsim,2);
lafilter_ll{1} = zeros(Tsim,1);
lafilter_ll{2} = zeros(Tsim,1);
lafilter_s     = zeros(Tsim,2);
lafilter_a{1}  = zeros(Tsim,2);
lafilter_a{2}  = zeros(Tsim,2);
lafilter_P{1}  = zeros(Tsim,2,2);
lafilter_P{2}  = zeros(Tsim,2,2);

x1 = ones(1,m)/(eye(m)-PI+ones(m,m));
x1 = x1';

for it=1:Tsim
    
    yts = simy(it,:)';
    x0 = x1;

    % solution step:
    xcur = [atp1{1} atp1{2}]*x1;
    rbccoef = rbcar1mc_tpcoef(xcur,B0,pvec);
    coefs(it,:,:) = rbccoef;
    for s = 1:2
        at{s} = atp1{s};
        Pt{s} = Ptp1{s};

        % read/update the system matrices
        Zt = [alf        1; 
              rbccoef(s,2:3)]; 

        dt = [0; rbccoef(s,1)];

        Ht = H; 
        Qt = sigz(s);

        ct = [rbccoef(s,4); 0]; 
        Tt = [rbccoef(s,5:6); 
              0      rhoz(s)]; 
        Rt = [0; 1];
    
        % Kalman filter equations:
        vt      = yts - Zt*at{s} - dt;
        Ft      = Zt*Pt{s}*Zt' + Ht;
        att{s}  = at{s} + (Pt{s}*Zt')/Ft*vt;
        atp1{s} = Tt*att{s} + ct;
        Kt      = Tt*Pt{s}*Zt'/Ft;
        Ptp1{s} = Tt*Pt{s}*(Tt-Kt*Zt)' + Rt*Qt*Rt';
        if it<Tsim
            lafilter_a{s}(it+1,:)    = atp1{s};
            lafilter_P{s}(it+1,:,:)  = Ptp1{s};
        end
        lafilter_y{s}(it,:)  = Zt*att{s} + dt; 
        lafilter_ll{s}(it) = (mvnpdf(vt,0,Ft));
    end
    
    xp1 = PI*x0;
    lls = [(lafilter_ll{1}(it)); (lafilter_ll{2}(it))];
    x1  = xp1.*lls ./ sum(xp1.*lls); 
    lafilter_s(it,:) = x1;
    
end

% Compute Log Likelihood
logl = sum(log(lafilter_ll{1}(:).*lafilter_s(it,1) + lafilter_ll{2}(:).*lafilter_s(it,2)));

% Plot Filter Output

figure(1); tiledlayout('flow');
nexttile;
plot(lafilter_s(:,1),'LineWidth',2); hold on;
plot(1-st,'*');
title("Regime",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','FontSize',16)
nexttile;
plot(sum([lafilter_a{1}(2:end,1) lafilter_a{2}(2:end,1)].*lafilter_s(2:end,:),2),'LineWidth',2);
hold on;
plot(kt(2:end),'--','LineWidth',2);
title("k",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','FontSize',16)
nexttile;
plot(sum([lafilter_a{1}(2:end,2) lafilter_a{2}(2:end,2)].*lafilter_s(2:end,:),2),'LineWidth',2);
hold on;
plot(zt(1:end-1),'--','LineWidth',2);
title("z",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','FontSize',16)
legend({"Filtered","True"},'interpreter','latex','fontsize',16,'box','off','orientation','horizontal','location','southoutside')

figure(2); tiledlayout('flow')
nexttile;
plot(sum([lafilter_y{1}(2:end,1) lafilter_y{2}(2:end,1)].*lafilter_s(2:end,:),2),'LineWidth',2);
hold on;
plot(simy(2:end,1),'--','LineWidth',2); title("y",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','FontSize',16)

nexttile;
plot(sum([lafilter_y{1}(2:end,2) lafilter_y{2}(2:end,2)].*lafilter_s(2:end,:),2),'LineWidth',2);
hold on;
plot(simy(2:end,2),'--','LineWidth',2); title("c",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','FontSize',16)
legend({"Fitted","True"},'interpreter','latex','fontsize',16,'box','off','orientation','horizontal','location','southoutside')