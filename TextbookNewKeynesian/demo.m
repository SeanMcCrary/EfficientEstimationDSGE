%% Textbook New Keynesian model: filtering demo
% Eva F. Janssens and Sean McCrary
% "Efficient Estimation of Nonlinear DSGE Models"

clear; clc; close all;
root_dir = fileparts(mfilename('fullpath'));

%% Fixed calibration from TextbookNK/main_filter_zlb.m
params.tau    = 1;
params.beta   = 0.96^(1/4);
params.eta    = 2;
params.kappa  = 0.07;
params.phi_pi = 2;
params.phi_y  = 0.5;
params.rho_z = 0.85;
params.rho_d = 0.92;
params.rho_m = 0.70;
params.sig_z = 0.005;
params.sig_d = 0.012;
params.sig_m = 0.007;
params.sigMEn  = 0.002; % first observable is output
params.sigMEpi = 0.006;
params.sigMEr  = 0.001;

%% Data
data = load(fullfile(root_dir,'demo_data.mat'),'y_obs','pi_obs','r_obs');
Y_observed = [data.y_obs data.pi_obs data.r_obs];
years = 1966+(0:size(Y_observed,1)-1)'/4;

% Reference inflation and nominal rate, with beta = exp(pi_bar-r_bar).
pi_bar = mean(data.pi_obs);
r_bar  = pi_bar-log(params.beta);
offset = [0 pi_bar r_bar];
Y = Y_observed-offset;
params.Rss  = exp(r_bar);
params.rZLB = r_bar; % zero nominal rate is a deviation of -r_bar
options.first_observable = 'output';
options.dp_horizon = 12;
options.binding_indicator = data.r_obs<=0; % all false in this sample

%% Filtering
methods = {'loglinear','tp_no_zlb','dp_no_zlb','dp_zlb'};
labels  = {'Log-linear + KF','TP1 + TVKF','DP1 + TVKF','DP1-ZLB + TVKF'};
n_methods = numel(methods);
fits = cell(1,n_methods);
loglik = zeros(n_methods,1);
seconds = zeros(n_methods,1);
for j = 1:n_methods
    timer = tic;
    fits{j} = nk_filter_tvkf(Y,params,methods{j},options);
    seconds(j) = toc(timer);
    loglik(j) = fits{j}.loglik;
end
comparison = table(labels',loglik,seconds, ...
    'VariableNames',{'Method','LogLikelihood','Seconds'});
disp(comparison);

%% Filtered states and observables
figure('Color','w','Position',[100 100 1450 750]);
layout = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
colors = [0.85 0.45 0.05; 0.00 0.50 0.32; 0.75 0.12 0.12; 0.10 0.35 0.80];
styles = {'-','--',':','-.'};
state_labels = {'Technology','Preference','Monetary policy'};
observable_labels = {'Output','Inflation','Nominal rate'};
for k = 1:3
    nexttile; hold on;
    for j = 1:n_methods
        plot(years,fits{j}.filtered_state(:,k),styles{j}, ...
            'Color',colors(j,:),'LineWidth',1.3,'DisplayName',labels{j});
    end
    title(['Filtered ' lower(state_labels{k}) ' state']);
    xlabel('Year'); grid on; xlim(years([1 end]));
    if k==1; legend('Location','best'); end
end
for k = 1:3
    nexttile; hold on;
    plot(years,Y_observed(:,k),'k','LineWidth',1.2,'DisplayName','Observed');
    for j = 1:n_methods
        plot(years,fits{j}.filtered_observables(:,k)+offset(k),styles{j}, ...
            'Color',colors(j,:),'LineWidth',1.1,'DisplayName',labels{j});
    end
    title(observable_labels{k}); xlabel('Year');
    grid on; xlim(years([1 end]));
    if k==1; legend('Location','best'); end
end
title(layout,'Textbook New Keynesian model: 1966:Q1-2007:Q4');
