clc; clear; close all; 

% Stochastic Growth Model with productivity following stochastic volatility 
% Use EKF to filter
% Authors: Eva Janssens & Sean McCrary
% Date: 3rd of June 2025

rng(1994); 

% parameters stochastic volatility
rz = 0.95; 
mu = -12.5; 
rx = 0.975; 
sx = 0.02; 

tau = 6; % has to be high for volatility to matter!  
bet = 1.04^(-1/4); 
alf = 0.36; 
del = 0.05; 

[nodes, weights] = normal_gh_quadrature(11); % Gauss-Hermite quadrature

% collect all parameters
pvec = [rz;mu;rx;sx;tau;bet;alf;del;nodes;weights']; 

% solve model at the steady state
B0   = zeros(8,1);
xt   = [0.00;0;mu]; 
fun  = @(b) Res_eval(xt,b,pvec); 

options = optimoptions('fsolve', ...
                       'Display', 'none', ...
                       'FunctionTolerance', 1e-14, ...
                       'StepTolerance', 1e-14, ...
                       'MaxIterations', 2000, ...
                       'MaxFunctionEvaluations', 20000);

Bss   = fsolve(fun,B0,options); 
Bss2  = rbcar1sv_tpcoef(xt,Bss,pvec); 

%% simulate a time series 
T      = 500; 
burn   = 250; 
st     = mu*ones(T,1); 
zt     = zeros(T,1); 
epst   = normrnd(0,1,T,1); 
etat   = normrnd(0,1,T,1); 

% simulate stochastic process
for ti=2:T
    st(ti) = (1-rx)*mu + rx*st(ti-1) + sx*etat(ti); 
    zt(ti) = rz*zt(ti-1) + exp(st(ti)/2)*epst(ti);
end 

kt      = zeros(T,1); 
Bt      = zeros(T,8); 
Bt(1,:) = Bss; 
ct      = zeros(T,1); 
yt      = zeros(T,1); 

% simulate the endogenous variables
for ti=2:T

    % solve at next point
    xt       = [kt(ti-1);zt(ti);st(ti)];  
    Btemp    = rbcar1sv_tpcoef(xt,Bss,pvec); 
    Bt(ti,:) = Btemp'; 

    % observables follow from the policy coefficients
    kt(ti) = Bt(ti,5) + Bt(ti,6)*kt(ti-1) + Bt(ti,7)*zt(ti) + Bt(ti,8)*st(ti);
    ct(ti) = Bt(ti,1) + Bt(ti,2)*kt(ti-1) + Bt(ti,3)*zt(ti) + Bt(ti,4)*st(ti);
    yt(ti) = zt(ti) + alf*kt(ti-1); 

end 

% remove burnin 
cts = ct(burn+1:end); 
yt  = yt(burn+1:end); 
zt  = zt(burn+1:end); 
st  = st(burn+1:end); 
kt  = kt(burn:end-1); 

% Collect observables and add measurement error

simy = [yt cts]';
xsim = [kt zt st];
Tsim = length(simy);

sigmey = 0.0005*std(simy(1,:));
sigmec = 0.0001*std(simy(2,:));
H = diag([sigmey^2 sigmec^2]);
simy = simy + sqrt(H)*randn(2,Tsim);
simy = simy';

sigu = 1;

% construct the system matrices that are not time-varying
Q = diag([sigu^2 sx^2]);

T = diag([rz rx]);

c = [0; (1-rx)*mu];

%% Extended Kalman filter

% Initalization:
lafilter_a    = zeros(Tsim,3);
lafilter_logl = zeros(Tsim,1);
lafilter_P    = zeros(Tsim,3,3);
lafilter_y    = zeros(Tsim,2);

atp1   = [0; 0; mu]; 
att    = atp1;
Ptp1   = ([std(simy(1,:))^2 0 0 ; 0 exp(mu)/(1-rz^2) 0 ; 0 0 sx^2/(1-rx^2)]);

for it=1:Tsim

    % advance time
    at = atp1;
    Pt = Ptp1;

    % solve the model
    fun      = @(b) Res_eval(at,b,pvec); 
    rbccoef  = rbcar1sv_tpcoef(at,Bss,pvec); 

    yt = simy(it,:)';

    % read/update the system matrices
    Zt = [alf        1   0; 
          rbccoef(2:4)']; 

    dt = [0; rbccoef(1)];

    Ht = H; 
    Qt = Q;

    ct = [rbccoef(5); 0; (1-rx)*mu]; 

    Tt = [rbccoef(6:end)'; 
          0      rz         0;
          0       0        rx]; 

    Rt = [0 0; exp(atp1(3)/2) 0; 0  1];

    % Kalman filter equations:
    vt   = yt - Zt*at - dt;
    Ft   = Zt*Pt*Zt' + Ht;
    att  = at + (Pt*Zt')/Ft*vt;
    atp1 = Tt*att + ct;
    Kt   = Tt*Pt*Zt'/Ft;
    Ptp1 = Tt*Pt*(Tt-Kt*Zt)' + Rt*Qt*Rt';
    if it<Tsim
        lafilter_a(it+1,:)    = atp1;
        lafilter_P(it+1,:,:)  = Ptp1;
    end
    lafilter_y(it,:)  = Zt*att + dt; 
    lafilter_logl(it) = log(mvnpdf(vt,0,Ft));
end

logl = sum(lafilter_logl);


%% Make plots

figure(4); tiledlayout(3,1);
nexttile;
plot(lafilter_a(2:end,3),'LineWidth',2); hold on;
plot(st(1:end),'--','LineWidth',2);
title("Volatility",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','FontSize',16)
nexttile;
plot(lafilter_a(2:end,1),'LineWidth',2);
hold on;
plot(kt(2:end),'--','LineWidth',2);
title("k",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','FontSize',16)
nexttile;
plot(lafilter_a(2:end,2),'LineWidth',2);
hold on;
plot(zt(1:end-1),'--','LineWidth',2);
title("z",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','FontSize',16)
legend({"Filtered","True"},'interpreter','latex','fontsize',16,'box','off','orientation','horizontal','location','southoutside')

figure(5); tiledlayout('flow')
nexttile;
plot(lafilter_y(:,1),'LineWidth',2);
hold on;
plot(simy(2:end,1),'--','LineWidth',2); title("y",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','FontSize',16)

nexttile;
plot(lafilter_y(:,2),'LineWidth',2);
hold on;
plot(simy(2:end,2),'--','LineWidth',2); title("c",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','FontSize',16)
legend({"Fitted","True"},'interpreter','latex','fontsize',16,'box','off','orientation','horizontal','location','southoutside')