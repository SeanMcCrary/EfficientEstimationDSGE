%--------------------------------------------------------------------------
% demo_filter_comparison.m
% (c) Eva F. Janssens and Sean McCrary
% These codes are part of the demo codes for the paper "Efficient Estimation 
% of Nonlinear DSGE Models" by Janssens & McCrary.
% 
% Simple comparison of alternative solution/filter combinations
% in the DMP model.
%
% One dataset is simulated from the global solution. The same data are
% then filtered using:
%
%   1. Log-linear solution + Kalman filter
%   2. Taylor Projection (TP1) + time-varying Kalman filter
%   3. Dynamic Perturbation (DP1) + time-varying Kalman filter
%   4. Global solution + extended Kalman filter
%   5. Global solution + particle filter
%
% The script plots true and filtered paths and reports normalized MSE
% and runtime for each method.
%--------------------------------------------------------------------------

clear; close all; clc;

script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir,'helpers'));

%% User choices

Tsim = 200;

% Global solution
Jglobal = 5000;
Nglobal = 24;

% Dynamic Perturbation
Ndp = 100;

% Particle filter
M = 2000;

% Reproducibility
base_seed = 1995;
pf_seed   = base_seed + 100000;

% Measurement-error standard deviation as a share of the
% unconditional standard deviation of each observable
measurement_error_share = 0.20;

% Save figure and compact summary table
save_outputs = false;


%% Parameters

par = dmp_parameters();

alfa  = par.alpha;
nu    = par.nu;
beta  = par.beta;
delta = par.delta;
eta   = par.eta;
rho   = par.rho;
sigma = par.sigma;


%% Global solution

% The global solution is used as the data-generating process.
globalDMP = GB_DMP(par,Jglobal,Nglobal);


%% Measurement error

% Compute the unconditional variation in the observables under the
% calibrated productivity distribution and set measurement error equal
% to a fixed share of that variation.

zcal = linspace(-6*par.sigz,6*par.sigz,4001)';

weight = exp(-0.5*(zcal/par.sigz).^2);
weight([1 end]) = 0.5*weight([1 end]);
weight = weight/sum(weight);

qcal     = EVAL_GB_DMP(globalDMP,zcal);
thetacal = -qcal/alfa;
wagecal  = WAGE_DMP(par,zcal);

mean_theta = sum(weight.*thetacal);
mean_wage  = sum(weight.*wagecal);

sdtheta = sqrt(sum(weight.*(thetacal-mean_theta).^2));
sdwage  = sqrt(sum(weight.*(wagecal-mean_wage).^2));

sigtheta = measurement_error_share*sdtheta;
sigwage  = measurement_error_share*sdwage;

% Parameter vector used by all filters:
%
% [alpha nu beta delta eta rho sigma sigtheta sigwage]
parvec = [ ...
    alfa ...
    nu ...
    beta ...
    delta ...
    eta ...
    rho ...
    sigma ...
    sigtheta ...
    sigwage];

% Variances used to normalize the MSEs
normalization_variance = [ ...
    sdtheta^2 ...
    sdwage^2 ...
    par.sigz^2];


%% Simulate one dataset from the global solution

rng(base_seed,'twister');

z0          = par.sigz*randn;
innovations = randn(Tsim-1,1);

sim = sim_global_dmp( ...
    par, ...
    globalDMP, ...
    Tsim, ...
    z0, ...
    innovations);

true_theta = sim.tightness;
true_wage  = sim.wage;
true_z     = sim.logtfp;

% Add measurement error to the observables used by the filters
obs_theta = true_theta + sigtheta*randn(Tsim,1);
obs_wage  = true_wage  + sigwage *randn(Tsim,1);

data = [obs_theta obs_wage];


%% Filter using alternative solution methods

%--------------------------------------------------------------------------
% Log-linear + KF
%--------------------------------------------------------------------------

tic;

[~,filter_z_KF,~,filter_y_KF] =  kfilter(parvec,data);

time_KF = toc;


%--------------------------------------------------------------------------
% TP1 + TVKF
%--------------------------------------------------------------------------

tic;

[~,filter_z_TP,~,filter_y_TP] =  tpfilter(parvec,data);

time_TP = toc;


%--------------------------------------------------------------------------
% DP1 + TVKF
%--------------------------------------------------------------------------

tic;

[~,filter_z_DP,~,filter_y_DP] =  dpfilter(parvec,data,Ndp);

time_DP = toc;


%--------------------------------------------------------------------------
% Global + EKF
%--------------------------------------------------------------------------

tic;

[~,filter_z_EKF,filter_y_EKF] =  EKfilter(parvec,data,Jglobal,Nglobal);

time_EKF = toc;


%--------------------------------------------------------------------------
% Global + PF
%--------------------------------------------------------------------------

tic;

[~,filter_z_PF,filter_y_PF] = pfilter(parvec, data, M, Jglobal, Nglobal, pf_seed);

time_PF = toc;


%% Filtering accuracy

accuracy_KF = filter_accuracy(filter_z_KF,filter_y_KF, ...
    true_z,true_theta,true_wage, normalization_variance);

accuracy_TP = filter_accuracy(filter_z_TP,filter_y_TP, ...
    true_z,true_theta,true_wage, normalization_variance);

accuracy_DP = filter_accuracy(filter_z_DP,filter_y_DP, ...
    true_z,true_theta,true_wage, normalization_variance);

accuracy_EKF = filter_accuracy(filter_z_EKF,filter_y_EKF, ...
    true_z,true_theta,true_wage, normalization_variance);

accuracy_PF = filter_accuracy(filter_z_PF,filter_y_PF, ...
    true_z,true_theta,true_wage, normalization_variance);


%% Compact summary table

method = { ...
    'Log-linear + KF'; ...
    'TP1 + TVKF'; ...
    'DP1 + TVKF'; ...
    'Global + EKF'; ...
    'Global + PF'};

accuracy = [ ...
    accuracy_KF; ...
    accuracy_TP; ...
    accuracy_DP; ...
    accuracy_EKF; ...
    accuracy_PF];

runtime = [ ...
    time_KF; ...
    time_TP; ...
    time_DP; ...
    time_EKF; ...
    time_PF];

% Normalized MSE x 100
mse_theta = 100*accuracy(:,1);
mse_wage  = 100*accuracy(:,2);
mse_z     = 100*accuracy(:,3);

summary_table = table( ...
    method, ...
    mse_theta, ...
    mse_wage, ...
    mse_z, ...
    runtime, ...
    'VariableNames',{ ...
        'Method', ...
        'MSE_theta', ...
        'MSE_wage', ...
        'MSE_z', ...
        'Seconds'});

% Compact console output
fprintf('\n');
fprintf('Filtering comparison (based on a single run)\n');
fprintf('--------------------------------------------------------------------------\n');
fprintf('%-20s %11s %11s %11s %11s\n', ...
    'Method','MSE theta','MSE wage','MSE z','Seconds');
fprintf('--------------------------------------------------------------------------\n');

for i = 1:height(summary_table)
    fprintf('%-20s %11.3f %11.3f %11.3f %11.3f\n', ...
        summary_table.Method{i}, ...
        summary_table.MSE_theta(i), ...
        summary_table.MSE_wage(i), ...
        summary_table.MSE_z(i), ...
        summary_table.Seconds(i));
end

fprintf('--------------------------------------------------------------------------\n\n');


%% Figure: true versus filtered paths

t = (1:Tsim)';

fig = figure('Color','w', 'Position',[100 100 1200 900]);

layout = tiledlayout(fig, 3,1, 'TileSpacing','compact', 'Padding','compact');


%--------------------------------------------------------------------------
% Productivity
%--------------------------------------------------------------------------

ax1 = nexttile(layout);

hold(ax1,'on');

pTrue = plot(ax1,t,true_z, 'k-', 'LineWidth',2.5);
pKF = plot(ax1,t,filter_z_KF, '-.', 'LineWidth',1.8);
pTP = plot(ax1,t,filter_z_TP, '--', 'LineWidth',1.8);
pDP = plot(ax1,t,filter_z_DP, '-', 'LineWidth',1.8);
pEKF = plot(ax1,t,filter_z_EKF, ':', 'LineWidth',2.0);
pPF = plot(ax1,t,filter_z_PF, '--', 'LineWidth',1.8);

hold(ax1,'off');

title(ax1,'Productivity state $z_t$', 'Interpreter','latex');

ylabel(ax1,'Log productivity', 'Interpreter','latex');

legend(ax1, ...
    [pTrue pKF pTP pDP pEKF pPF], ...
    {'True', ...
     'Log-linear + KF', ...
     'TP1 + TVKF', ...
     'DP1 + TVKF', ...
     'Global + EKF', ...
     'Global + PF'}, ...
    'Interpreter','latex', ...
    'NumColumns',3, ...
    'Box','off', ...
    'Location','best');

set(ax1, 'TickLabelInterpreter','latex', 'FontSize',14, 'LineWidth',1);

xlim(ax1,[1 Tsim]);
box(ax1,'off');


%--------------------------------------------------------------------------
% Market tightness
%--------------------------------------------------------------------------

ax2 = nexttile(layout);
hold(ax2,'on');
plot(ax2,t,true_theta, 'k-', 'LineWidth',2.5);
plot(ax2,t,filter_y_KF(:,1), '-.', 'LineWidth',1.8);
plot(ax2,t,filter_y_TP(:,1), '--', 'LineWidth',1.8);
plot(ax2,t,filter_y_DP(:,1), '-', 'LineWidth',1.8);
plot(ax2,t,filter_y_EKF(:,1), ':', 'LineWidth',2.0);
plot(ax2,t,filter_y_PF(:,1), '--', 'LineWidth',1.8);

hold(ax2,'off');

title(ax2,'Market tightness', 'Interpreter','latex');
ylabel(ax2,'$\theta_t$', 'Interpreter','latex');

set(ax2, 'TickLabelInterpreter','latex', 'FontSize',14, 'LineWidth',1);

xlim(ax2,[1 Tsim]);
box(ax2,'off');


%--------------------------------------------------------------------------
% Wage
%--------------------------------------------------------------------------

ax3 = nexttile(layout);

hold(ax3,'on');

plot(ax3,t,true_wage, 'k-', 'LineWidth',2.5);
plot(ax3,t,filter_y_KF(:,2), '-.', 'LineWidth',1.8);
plot(ax3,t,filter_y_TP(:,2), '--', 'LineWidth',1.8);
plot(ax3,t,filter_y_DP(:,2), '-', 'LineWidth',1.8);
plot(ax3,t,filter_y_EKF(:,2), ':', 'LineWidth',2.0);
plot(ax3,t,filter_y_PF(:,2), '--', 'LineWidth',1.8);

hold(ax3,'off');

title(ax3,'Wage','Interpreter','latex');
xlabel(ax3,'Time', 'Interpreter','latex');
ylabel(ax3,'$w_t$','Interpreter','latex');
set(ax3, 'TickLabelInterpreter','latex', 'FontSize',14, 'LineWidth',1);

xlim(ax3,[1 Tsim]);
box(ax3,'off');


%% Save outputs

if save_outputs

    output_dir = fullfile(script_dir,'output');

    if exist(output_dir,'dir') ~= 7
        mkdir(output_dir);
    end

    exportgraphics( ...
        fig, ...
        fullfile(output_dir,'demo_filter_comparison.pdf'), ...
        'ContentType','vector');

    writetable( ...
        summary_table, ...
        fullfile(output_dir,'demo_filter_comparison.csv'));

end


%% Local function

function accuracy = filter_accuracy( ...
    filter_z,filter_y, ...
    true_z,true_theta,true_wage, ...
    normalization_variance)

error_theta = filter_y(:,1)-true_theta;
error_wage  = filter_y(:,2)-true_wage;
error_z     = filter_z-true_z;

% Normalized mean squared errors
accuracy = [ ...
    mean(error_theta.^2)/normalization_variance(1) ...
    mean(error_wage.^2)/normalization_variance(2) ...
    mean(error_z.^2)/normalization_variance(3)];

end