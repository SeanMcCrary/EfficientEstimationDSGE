%% NKDMP filtering demo
% Run in MATLAB R2025b with Statistics and Machine Learning Toolbox.
% Each data file contains seven observed series, dates, and fixed parameters.
% Pre-ZLB parameters: first_try_KF.mat, pardraws (same for all three filters).
% Full-sample parameters: dp_zlb_sampler_results_fullsample.mat, results.finalPar.
% The figures plot yhat = E[y_t | Y_{1:t}], in 100 x the input log units.
% The prepared n_obs series measures hours; no data are transformed here.
clear; close all; clc;
demoDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(demoDir,'helpers')));

names = {'GDP deflator inflation','CPI inflation','Labor input (n)', ...
         'Market tightness','Nominal interest rate','Separation rate','Job-finding rate'};
warningState = warning;

%% Pre-ZLB sample: 1984Q1-2007Q4
pre = load(fullfile(demoDir,'demo_data_prezlb.mat'));
Ypre = [pre.pi_obs, pre.pi_obs2, pre.n_obs, pre.tg_obs, ...
        pre.r_obs, pre.s_obs, pre.jf_obs];
par = pre.par;  % Same fixed parameters for all three methods.

[~,~,~,y_loglin,states_loglin] = kfilter(par,Ypre);
[~,~,~,y_dp,states_dp] = dpfilter(par,Ypre);
[~,~,~,y_tp,states_tp] = tpfilter(par,Ypre);
warning(warningState);

figure('Name','NKDMP: pre-ZLB sample','Color','w','Position',[100 100 1200 700]);
tiledlayout(2,4,'TileSpacing','compact','Padding','compact');
for j=1:7
    nexttile;
    plot(pre.dates,100*Ypre(:,j),'Color',[0.55 0.55 0.55],'LineWidth',1); hold on;
    plot(pre.dates,100*y_loglin(:,j),'-','Color',[0 0.45 0.74],'LineWidth',1.3);
    plot(pre.dates,100*y_dp(:,j),'--','Color',[0.85 0.33 0.10],'LineWidth',1.3);
    plot(pre.dates,100*y_tp(:,j),':','Color',[0.47 0.67 0.19],'LineWidth',1.5);
    title(names{j}); grid on; axis tight;
    ylabel('100 x log units'); xtickformat('yyyy');
    if j==1
        lgd=legend('Observed','Log-linear','DP','TP','Orientation','horizontal');
        lgd.Layout.Tile='south';
    end
end
sgtitle('NKDMP: filtered observables, 1984Q1-2007Q4');

%% Sample including the ZLB: 1984Q1-2019Q4
zlb = load(fullfile(demoDir,'demo_data_zlb.mat'));
Yzlb = [zlb.pi_obs, zlb.pi_obs2, zlb.n_obs, zlb.tg_obs, ...
        zlb.r_obs, zlb.s_obs, zlb.jf_obs];
par = zlb.par;
bindZLB = Yzlb(:,5) < (1.005^(1/4)-1);  % Cutoff in the original estimation script.

[~,~,~,y_dp_zlb,states_dp_zlb] = dp_zlb_filter(par,Yzlb,bindZLB);

figure('Name','NKDMP: sample including the ZLB','Color','w','Position',[120 120 1200 700]);
tiledlayout(2,4,'TileSpacing','compact','Padding','compact');
for j=1:7
    nexttile;
    plot(zlb.dates,100*Yzlb(:,j),'Color',[0.55 0.55 0.55],'LineWidth',1); hold on;
    plot(zlb.dates,100*y_dp_zlb(:,j),'-','Color',[0.49 0.18 0.56],'LineWidth',1.3);
    title(names{j}); grid on; axis tight;
    ylabel('100 x log units'); xtickformat('yyyy');
    if j==1
        lgd=legend('Observed','DP-ZLB','Orientation','horizontal');
        lgd.Layout.Tile='south';
    end
end
sgtitle('NKDMP: filtered observables, 1984Q1-2019Q4');
