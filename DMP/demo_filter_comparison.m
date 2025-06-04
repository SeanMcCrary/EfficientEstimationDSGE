% This script simulates from the global solution of the DMP model and
% estimates using four solution/filtering methods:
% (1) Taylor Projection + Local Approximation Filter (TV-KF)
% (2) Log linearization + Kalman Filter
% (3) Global model solution + Particle Filter
% (4) Global model solution + Extended Kalman Filter

% Authors: Eva Janssens & Sean McCrary
% Date: 3rd of June 2025

% preamble:
clear; close all; clc;
rng(1995);

%% Step 0: simulate from the Global solution of the DMP model

r     = 0.04;                % interest rate
bet   = (1/(1+r))^(1/4);     % discount factor (quarterly)
alfa  = 0.7;                 % matching function parameter
rhoz  = 0.985;               % tfp persistence
sige  = 0.0015;              % tfp innovation volatility
eta   = 0.50;                % bargaining weight
b     = 0.94;                % oc of employment
sigz  = sige/sqrt(1-rhoz^2); % tfp sd

nss  = 0.945;                                 % ss employment
qss  = 0.7;                                   % ss vacancy-filling
tgss = 1;                                     % ss market tightness
ass  = qss/((1+tgss^alfa)^(-1/alfa));         % ss match efficiency
del  = qss*tgss*(1-nss)/((1-qss*tgss)*nss);   % ss separation rate
kap  = (1-eta)*(1-b)*qss/(1-bet*(1-del));     % vacancy cost
c    = kap/ass;
uss  = 1-nss;

% log-linear coefficient
gll  = ((qss^(1-alfa))*(tgss^(-alfa)))*((1-eta)/c)*((1-bet*(1-del)*rhoz)^(-1))/ass^(1-alfa);

[~,epsi_nodes,weight_nodes] = Monomials_2(1,sige^2);

par.alfa  = alfa;
par.b     = b;
par.bet   = bet;
par.c     = c;
par.del   = del;
par.eta   = eta;
par.rhoz  = rhoz;
par.sige  = sige;
par.egrid = epsi_nodes;
par.ewgt  = weight_nodes;

% Solve the model so we can simulate from it
par.nsig    = 5;
poly_degree = 11; % degree of polynomial approximation 

dmp_global_coefs = dmp_global_coef(par,poly_degree);
p = poly_degree - 1;

Tsim = 200;

% Simulate tightness from global model given z0
% where z0 is initial TFP (z0=0 is steady state value)
% returns a structure: struct.tightness and struct.logtfp
% which are time series of length Tsim

rng(24)
z0   = normrnd(0,sigz,1);
sim_timeseries = sim_global_dmp(dmp_global_coefs,b,eta,rhoz,sige,Tsim,z0);

% "truth"
true_tightness  = sim_timeseries.tightness;
true_tfp        = sim_timeseries.logtfp;

% add measurement error
sigtg  = 0.2*std(true_tightness);
sigltg = 0.2*std(log(true_tightness));

obs_tightness  = true_tightness      + sigtg*randn(Tsim,1);
obs_ltightness = log(true_tightness) + sigltg*randn(Tsim,1);

varotg = var(obs_tightness);

%% Filter with TP
parvec = [alfa b bet c del eta rhoz sige sigtg];

tic
[logltpf,tpfilter_a,tpfilter_P,tpfilter_yhat] = tpfilter_TG(parvec,obs_tightness);
timeTVKF=toc; 

%% Filter with log-linearization and KF

tic
[loglkf,kfilter_a,kfilter_P,kfilter_yhat] = kfilter_TG(parvec,obs_ltightness);
timeKF = toc; 

%% Filter with EKF 

tic;
[loglekf,ekfilter_a,ekfilter_yhat] = EKfilter_TG(parvec,obs_tightness);
timeEKF = toc; 

%% Filter with particle filter

tic
particles = 20000;
[logp,pfilter_a,pfilter_yhat] = pfilter_TG(parvec,obs_tightness,particles);
timePF = toc; 

%% plot the filter output
figure(1); tiledlayout(2,1); nexttile;

plot(exp(true_tfp(2:end-1)), '-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]); hold on;
plot(exp(tpfilter_a(3:end)), 'g--', 'LineWidth', 2); hold on;
plot(exp(pfilter_a(3:end)), 'm:', 'LineWidth', 2);
plot(exp(kfilter_a(3:end)), 'r-.', 'LineWidth', 2);
plot(exp(ekfilter_a(3:end)), 'b--', 'LineWidth', 2);

title('Productivity','interpreter','latex','fontsize',14)
xlabel('Time','interpreter','latex','fontsize',14)
set(gca,'TickLabelInterpreter','latex','fontsize',14)

nexttile;
plot((true_tightness(1:end-1)), '-', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]); hold on;
plot((tpfilter_yhat(3:end)), 'g-', 'LineWidth', 2); hold on;
plot((pfilter_yhat(2:end)), 'm:', 'LineWidth', 2);
plot((ekfilter_yhat(3:end)), 'b--', 'LineWidth', 2);
plot((tgss+kfilter_yhat(2:end)), 'r-.', 'LineWidth', 2);
title('Tightness','interpreter','latex','fontsize',14)
xlabel('Time','interpreter','latex','fontsize',14)
set(gca,'TickLabelInterpreter','latex','fontsize',14)
legend({'True','TV-KF','PF','EKF','KF'},'interpreter','latex','fontsize',14,...
    'box','off','orientation','horizontal','location','southoutside')

% print runtimes 
fprintf('Runtimes (sec) by method: TV-KV %4.4f, PF %4.4f, EKF %4.4f, KF %4.4f \n',timeTVKF,timePF,timeEKF,timeKF)