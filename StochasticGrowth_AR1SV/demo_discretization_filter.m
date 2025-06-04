clc; clear; close all; 

% Stochastic Growth Model with productivity following stochastic volatility 
% Filter discretizes the stochastic volatility component and uses a
% Hamilton filter for that
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
    fun      = @(b) Res_eval(xt,b,pvec); 
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

% discretize the volatility process
M = 5;
[Grid, T] = mytauchen(mu*(1-rx),rx,sx,M);
Grid = Grid';
x0   = ones(1,M)/(eye(M)-T+ones(M,M));
x0   = x0';

% collect observables and add measurement error
simy = [yt cts]';
xsim = [kt zt st];
Tsim = length(simy);

sigmey = 0.0005*std(simy(1,:));
sigmec = 0.0001*std(simy(2,:));

H = diag([sigmey^2 sigmec^2]);
simy = simy + sqrt(H)*randn(2,Tsim);
simy = simy';

sigu = 1;

Q = sigu^2;

%% filter

% Initalization:
atp1 = cell(M,1);
at   = cell(M,1);
Ptp1 = cell(M,1);
Pt   = cell(M,1);
att  = cell(M,1);

lafilter_y  = cell(M,1);
lafilter_a  = cell(M,1);
lafilter_P  = cell(M,1);
lafilter_ll = cell(M,1);
lafilter_s  = zeros(Tsim,M);

for iim=1:M
    atp1{iim}   = [0; 0];
    at{iim}     = atp1{iim}; 
    Ptp1{iim}   = ([std(simy(1,:))^2 0  ; 0 exp(mu)/(1-rz^2)]); 
end

x1 = x0;

% Start filter
for it=1:Tsim

    acur = [];
    for idm=1:M
        acur = [acur at{idm}]; %#ok
    end
 
    % Compute the solution at the current state
    xcur = acur*x1;
    scur = Grid*x1;
    rbccoef    = rbcar1sv_tpcoef([xcur; scur],Bss,pvec);

    % do Kalman filter for every discrete level of volatility
    for iim=1:M

        x0 = x1;
        at{iim} = atp1{iim};
        Pt{iim} = Ptp1{iim};  

        yt = simy(it,:)';
    
        % read/update the system matrices
        Zt = [alf        1   ; 
              rbccoef(2:3)' ]; 
    
        dt = [0; rbccoef(1) + rbccoef(4)*Grid(iim)];
    
        Ht = H; 
        Qt = Q;
    
        ct = [rbccoef(5) + rbccoef(8)*Grid(iim); 0]; 
    
        Tt = [rbccoef(6:7)'; 
              0      rz    ]; 
    
        Rt = [0; exp(Grid(iim)/2)];
    
        % Kalman filter equations:
        vt   = yt - Zt*at{iim} - dt;
        Ft   = Zt*Pt{iim}*Zt' + Ht;
        att{iim}  = at{iim} + (Pt{iim}*Zt')/Ft*vt;
        atp1{iim} = Tt*att{iim} + ct;
        Kt   = Tt*Pt{iim}*Zt'/Ft;
        Ptp1{iim} = Tt*Pt{iim}*(Tt-Kt*Zt)' + Rt*Qt*Rt';
        if it<Tsim
            lafilter_a{iim}(it+1,:)    = atp1{iim};
            lafilter_P{iim}(it+1,:,:)  = Ptp1{iim};
        end
        lafilter_y{iim}(it,:)  = Zt*att{iim} + dt; 
        lafilter_ll{iim}(it) = (mvnpdf(vt,0,Ft));
    end

    % Hamilton filter step
    xp1 = T*x0;
    lls = zeros(M,1);
    for iim=1:M
        lls(iim) = lafilter_ll{iim}(it);
    end

    x1  = xp1.*lls ./ sum(xp1.*lls); 
    lafilter_s(it,:) = x1;
end

a1 = zeros(Tsim-1,M);
for iim=1:M
    a1(:,iim) = lafilter_a{iim}(2:end,1);
end
a2 = zeros(Tsim-1,M);
for iim=1:M
    a2(:,iim) = lafilter_a{iim}(2:end,2);
end
y1 = zeros(Tsim,M);
for iim=1:M
    y1(:,iim) = lafilter_y{iim}(1:end,1);
end

y2 = zeros(Tsim,M);
for iim=1:M
    y2(:,iim) = lafilter_y{iim}(1:end,2);
end

a1hat = sum(a1.*lafilter_s(2:end,:),2);
a2hat = sum(a2.*lafilter_s(2:end,:),2);
y1hat = sum(y1.*lafilter_s(1:end,:),2);
y2hat = sum(y2.*lafilter_s(1:end,:),2);

shat = sum(repelem(Grid,Tsim-1,1).*lafilter_s(2:end,:),2);

%% Make plots

figure(11); tiledlayout(3,1);
nexttile;
plot(shat(:,1),'LineWidth',2); hold on;
plot(st,'--','LineWidth',2);
title("Volatility",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','FontSize',16)
nexttile;
plot(a1hat,'LineWidth',2);
hold on;
plot(kt(2:end),'--','LineWidth',2);
title("k",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','FontSize',16)
nexttile;
plot(a2hat,'LineWidth',2);
hold on;
plot(zt(1:end-1),'--','LineWidth',2);
title("z",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','FontSize',16)
legend({"Filtered","True"},'interpreter','latex','fontsize',16,'box','off','orientation','horizontal','location','southoutside')

figure(21); tiledlayout('flow')
nexttile;
plot(y1hat,'LineWidth',2);
hold on;
plot(simy(2:end,1),'--','LineWidth',2); title("y",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','FontSize',16)

nexttile;
plot(y2hat,'LineWidth',2);
hold on;
plot(simy(2:end,2),'--','LineWidth',2); title("c",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','FontSize',16)
legend({"Fitted","True"},'interpreter','latex','fontsize',16,'box','off','orientation','horizontal','location','southoutside')