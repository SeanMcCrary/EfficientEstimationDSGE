clear; close all; clc;       % Clear workspace, close figures, clear command window
rng(1);                      % Set random seed for reproducibility

load("demo_data.mat")       % Load data

% Collect observed variables into a single matrix (T × 5)
Y = [pi_obs, n_obs, tg_obs, r_obs, jf_obs];

% MCMC sampling settings:
covscale = 0.1;    % Scale of proposal covariance matrix (tuned to target ~30% acceptance rate)
skip     = 5;      % Thinning interval: keep every 5th draw
burnIn   = 20;  % Number of initial draws to discard (burn-in period)
numDraws = 200; % Total number of MCMC draws (recommended: 100,000; runtime ≈ 2.5 hours)

% Initialize model parameters and steady-state values
bet    = exp(0.0067 - 0.0113);     % Discount factor (exp transform from estimated log parameter)
eta    = 0.5;                      % Bargaining weight 
gam    = 0.1;                      % Steady state markup 
alfa   = 0.5574;                   % Matching function parameter
nu     = 0.8;                      % Worker bargaining power or share of match surplus
nss    = 0.945;                    % Steady-state employment rate
qss    = 0.7;                      % Steady-state vacancy-filling probability
uss    = 1 - nss;                  % Steady-state unemployment rate
yss    = nss;                      % Steady-state output (normalized to equal employment)
tgss   = 1;                        % Steady-state labor market tightness (vacancy-unemployment ratio)
ass    = log(qss);                 % Steady-state log match efficiency
del    = qss * tgss * (1 - nss) / ((1 - qss * tgss) * nss);  % Steady-state separation rate
sss    = log(del);                 % Log of steady-state separation rate
c      = (1 - eta) * (1 - gam - nu) * qss / (1 - bet * (1 - del));  % Vacancy posting cost (from free-entry)
piss   = 0.008;                    % Steady-state inflation
rss    = 0.013;                    % Steady-state nominal interest rate
tau    = 2;                        % Risk-aversion 
lambda = 200;                      % Parameter for the softmax ZLB

% Order of parameters expected by nkdmp_zlb_tpcoef
% p      = [rz, sz, rd, sd, rm, sm, rs, ss, ra, sa, tau, tp, ty, kap, bet, alf, qss, eta, c, gam, del, nss, nu, rss, lam]
% pindex = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  11, 12, 13,  14,  15,  16,  17,  18, 19,  20,  21,  22,  23, 24,  25]

p = zeros(31,1);    % Initialize parameter vector

% Exogenous shock processes: AR(1) coefficients and standard deviations
p(1)  = 0.9420;     % rz:  TFP persistence (rho_z)
p(2)  = 0.001;      % sz:  Std. dev. of TFP shocks (sigma_z)
p(3)  = 0.9039;     % rd:  Discount rate shock persistence (rho_d)
p(4)  = 0.02;       % sd:  Std. dev. of discount shocks (sigma_d)
p(5)  = 0.7;        % rm:  Monetary policy shock persistence (rho_m)
p(6)  = 0.004;      % sm:  Std. dev. of monetary shocks (sigma_m)
p(7)  = 0.5188;     % rs:  Separation shock persistence (rho_s)
p(8)  = 0.01;       % ss:  Std. dev. of separation shocks (sigma_s)
p(9)  = 0.8360;     % ra:  Matching efficiency shock persistence (rho_a)
p(10) = 0.004;      % sa:  Std. dev. of matching efficiency shocks (sigma_a)

% Structural model parameters
p(11) = tau;        % tau:  Risk aversion 
p(12) = 4;          % tp:   Taylor rule coefficient on inflation (psi_pi)
p(13) = 0.15;       % ty:   Taylor rule coefficient on output (psi_y)
p(14) = 0.07;       % kap:  Phillips curve slope (kappa)
p(15) = bet;        % bet:  Discount factor
p(16) = alfa;       % alf:  Matching function 
p(17) = qss;        % qss:  Steady-state vacancy-filling rate 
p(18) = eta;        % eta:  Bargaining weight 
p(19) = c;          % c:    Vacancy posting cost
p(20) = gam;        % gam:  Steady state markup 
p(21) = del;        % del:  Separation rate
p(22) = nss;        % nss:  Steady-state employment rate
p(23) = nu;         % nu:   Outside option 
p(24) = rss;        % rss:  Steady-state nominal interest rate
p(25) = lambda;     % lam:  (ZLB constraint)


% Measurement error standard deviations (initial guess assumed to be 30% of the data's std dev)
p(26) = 0.3*std(pi_obs);   % Measurement error in inflation
p(27) = 0.3*std(n_obs);    % Measurement error in employment
p(28) = 0.3*std(tg_obs);   % Measurement error in market tightness
p(29) = 0.3*std(r_obs);    % Measurement error in interest rate
p(30) = 0.3*std(jf_obs);   % Measurement error in job finding rate

% Steady-state inflation 
p(31) = piss;               

% Final parameter vector
parvec = p;

% Run TP first time
logl = tpfilter_zlb(p,Y);

%% Specify all priors

% Prior for nu (bargaining power), truncated Beta on [0, truncH]
nuMean  = 0.9; nuStd = 0.025;
nuA     = nuMean*(nuMean*(1-nuMean)/(nuStd^2)-1);
nuB     = (1-nuMean)*(nuMean*(1-nuMean)/(nuStd^2)-1);
truncH  = 0.985;
nuPrior = @(nu) (nu<=truncH).*betapdf(nu, nuA, nuB)/(betacdf(truncH, nuA, nuB));

% Prior for kappa (Phillips curve slope), truncated Beta on [truncL, 1]
kpMean  = 0.05; kpStd = 0.02;
kpA     = kpMean*(kpMean*(1-kpMean)/(kpStd^2)-1);
kpB     = (1-kpMean)*(kpMean*(1-kpMean)/(kpStd^2)-1);
truncL  = 0.02;
kpPrior = @(kp) (kp>=truncL).*betapdf(kp,kpA,kpB)/(1-betacdf(truncL,kpA,kpB));

% Prior for alpha (matching function elasticity), standard Beta
alfaMean  = 0.5; alfaStd = 0.1;
alfaA     = alfaMean*(alfaMean*(1-alfaMean)/(alfaStd^2)-1);
alfaB     = (1-alfaMean)*(alfaMean*(1-alfaMean)/(alfaStd^2)-1);
alfaPrior = @(alfa) betapdf(alfa, alfaA, alfaB); 

% Prior for psi1 (Taylor rule coefficient on inflation), Gamma
psi1Mean = 1.5; psi1Std = 0.25;
psi1Scale = (psi1Std^2)/psi1Mean;
psi1Shape = psi1Mean/psi1Scale;
psi1Prior = @(psi1) gampdf(psi1,psi1Shape,psi1Scale);

% Prior for psi2 (Taylor rule coefficient on output), Gamma
psi2Mean = 0.25; psi2Std = 0.1;
psi2Scale = (psi2Std^2)/psi2Mean;
psi2Shape = psi2Mean/psi2Scale;
psi2Prior = @(psi2) gampdf(psi2,psi2Shape,psi2Scale);

% Inverse-Gamma-type priors for standard deviations of structural shocks
sigZmean = 0.02; sigZStd = 0.05;
sigZShockA = (sigZmean^2)/(sigZStd^2) + 2;
sigZShockB = sigZmean*((sigZmean^2)/(sigZStd^2) + 1);
sigZPrior = @(x) (sigZShockB^sigZShockA)/gamma(sigZShockA)*x.^(-sigZShockA-1).*exp(-sigZShockB./x);

sigDmean = 0.05; sigDStd = 0.1;
sigDShockA = (sigDmean^2)/(sigDStd^2) + 2;
sigDShockB = sigDmean*((sigDmean^2)/(sigDStd^2) + 1);
sigDPrior = @(x) (sigDShockB^sigDShockA)/gamma(sigDShockA)*x.^(-sigDShockA-1).*exp(-sigDShockB./x);

sigSmean = 0.05; sigSStd = 0.1;
sigSShockA = (sigSmean^2)/(sigSStd^2) + 2;
sigSShockB = sigSmean*((sigSmean^2)/(sigSStd^2) + 1);
sigSPrior = @(x) (sigSShockB^sigSShockA)/gamma(sigSShockA)*x.^(-sigSShockA-1).*exp(-sigSShockB./x);

sigmmean = 0.02; sigmStd = 0.05;
sigmShockA = (sigmmean^2)/(sigmStd^2) + 2;
sigmShockB = sigmmean*((sigmmean^2)/(sigmStd^2) + 1);
sigmPrior = @(x) (sigmShockB^sigmShockA)/gamma(sigmShockA)*x.^(-sigmShockA-1).*exp(-sigmShockB./x);

sigXmean = 0.02; sigXStd = 0.05;
sigXShockA = (sigXmean^2)/(sigXStd^2) + 2;
sigXShockB = sigXmean*((sigXmean^2)/(sigXStd^2) + 1);
sigXPrior = @(x) (sigXShockB^sigXShockA)/gamma(sigXShockA)*x.^(-sigXShockA-1).*exp(-sigXShockB./x);

% Beta priors for AR(1) persistence parameters
rhozMean = 0.5; rhozStd = 0.1;
rhozA = rhozMean*(rhozMean*(1-rhozMean)/(rhozStd^2)-1);
rhozB = (1-rhozMean)*(rhozMean*(1-rhozMean)/(rhozStd^2)-1);
rhozPrior = @(rhoz) betapdf(rhoz,rhozA,rhozB);

rhodMean = 0.5; rhodStd = 0.1;
rhodA = rhodMean*(rhodMean*(1-rhodMean)/(rhodStd^2)-1);
rhodB = (1-rhodMean)*(rhodMean*(1-rhodMean)/(rhodStd^2)-1);
rhodPrior = @(rhod) betapdf(rhod,rhodA,rhodB);

rhoSMean = 0.5; rhoSStd = 0.1;
rhoSA = rhoSMean*(rhoSMean*(1-rhoSMean)/(rhoSStd^2)-1);
rhoSB = (1-rhoSMean)*(rhoSMean*(1-rhoSMean)/(rhoSStd^2)-1);
rhoSPrior = @(rhoS) betapdf(rhoS, rhoSA, rhoSB);

rhoRMean = 0.5; rhoRStd = 0.1;
rhoRA = rhoRMean*(rhoRMean*(1-rhoRMean)/(rhoRStd^2)-1);
rhoRB = (1-rhoRMean)*(rhoRMean*(1-rhoRMean)/(rhoRStd^2)-1);
rhomPrior = @(rhoR) betapdf(rhoR, rhoRA, rhoRB);

rhoXMean = 0.5; rhoXStd = 0.1;
rhoXA = rhoXMean*(rhoXMean*(1-rhoXMean)/(rhoXStd^2)-1);
rhoXB = (1-rhoXMean)*(rhoXMean*(1-rhoXMean)/(rhoXStd^2)-1);
rhoXPrior = @(rhoX) betapdf(rhoX, rhoXA, rhoXB);

% Inverse-Gamma-type priors for measurement error standard deviations
sigmapi_obsMean = 0.5*0.01; sigmapi_obsStd = 0.05;
sigmapi_obsShockA = (sigmapi_obsMean^2)/(sigmapi_obsStd^2) + 2;
sigmapi_obsShockB = sigmapi_obsMean*((sigmapi_obsMean^2)/(sigmapi_obsStd^2) + 1);
sigmapi_obsPrior = @(x) (sigmapi_obsShockB^sigmapi_obsShockA)/gamma(sigmapi_obsShockA)*x.^(-sigmapi_obsShockA-1).*exp(-sigmapi_obsShockB./x);

sigman_obsMean = 0.5*0.01; sigman_obsStd = 0.05;
sigman_obsShockA = (sigman_obsMean^2)/(sigman_obsStd^2) + 2;
sigman_obsShockB = sigman_obsMean*((sigman_obsMean^2)/(sigman_obsStd^2) + 1);
sigman_obsPrior = @(x) (sigman_obsShockB^sigman_obsShockA)/gamma(sigman_obsShockA)*x.^(-sigman_obsShockA-1).*exp(-sigman_obsShockB./x);

sigmatg_obsMean = 0.5*0.03; sigmatg_obsStd = 0.05;
sigmatg_obsShockA = (sigmatg_obsMean^2)/(sigmatg_obsStd^2) + 2;
sigmatg_obsShockB = sigmatg_obsMean*((sigmatg_obsMean^2)/(sigmatg_obsStd^2) + 1);
sigmatg_obsPrior = @(x) (sigmatg_obsShockB^sigmatg_obsShockA)/gamma(sigmatg_obsShockA)*x.^(-sigmatg_obsShockA-1).*exp(-sigmatg_obsShockB./x);

sigmar_obsMean = 0.5*0.01; sigmar_obsStd = 0.05;
sigmar_obsShockA = (sigmar_obsMean^2)/(sigmar_obsStd^2) + 2;
sigmar_obsShockB = sigmar_obsMean*((sigmar_obsMean^2)/(sigmar_obsStd^2) + 1);
sigmar_obsPrior = @(x) (sigmar_obsShockB^sigmar_obsShockA)/gamma(sigmar_obsShockA)*x.^(-sigmar_obsShockA-1).*exp(-sigmar_obsShockB./x);

sigmajf_obsMean = 0.5*0.03; sigmajf_obsStd = 0.05;
sigmajf_obsShockA = (sigmajf_obsMean^2)/(sigmajf_obsStd^2) + 2;
sigmajf_obsShockB = sigmajf_obsMean*((sigmajf_obsMean^2)/(sigmajf_obsStd^2) + 1);
sigmajf_obsPrior = @(x) (sigmajf_obsShockB^sigmajf_obsShockA)/gamma(sigmajf_obsShockA)*x.^(-sigmajf_obsShockA-1).*exp(-sigmajf_obsShockB./x);

% Inverse-Gamma-type priors for steady-state inflation and interest rate
pissmean = 0.008; pissStd = 0.02;
pissShockA = (pissmean^2)/(pissStd^2) + 2;
pissShockB = pissmean*((pissmean^2)/(pissStd^2) + 1);
pissPrior = @(x) (pissShockB^pissShockA)/gamma(pissShockA)*x.^(-pissShockA-1).*exp(-pissShockB./x);

rssmean = 0.013; rssStd = 0.02;
rssShockA = (rssmean^2)/(rssStd^2) + 2;
rssShockB = rssmean*((rssmean^2)/(rssStd^2) + 1);
rssPrior = @(x) (rssShockB^rssShockA)/gamma(rssShockA)*x.^(-rssShockA-1).*exp(-rssShockB./x);

% Combine all individual priors into a joint prior density function
modelPrior = @(x) nuPrior(x(1)).*... 
                  kpPrior(x(2)).*alfaPrior(x(3)).*psi1Prior(x(4)).*...
                  psi2Prior(x(5)).*sigZPrior(x(6)).*sigDPrior(x(7)).*...
                  sigSPrior(x(8)).*sigmPrior(x(9)).*sigXPrior(x(10)).*...
                  rhozPrior(x(11)).*rhodPrior(x(12)).*rhoSPrior(x(13)).*...
                  rhomPrior(x(14)).*rhoXPrior(x(15)).*sigmapi_obsPrior(x(16)).*...
                  sigman_obsPrior(x(17)).*sigmatg_obsPrior(x(18)).*sigmar_obsPrior(x(19)).*...
                  sigmajf_obsPrior(x(20)).*pissPrior(x(21)).*rssPrior(x(22));

%% Initialize and run Metropolis-Hastings algorithm

% parvec = [rz1, sz2, rd3, sd4, rm5, sm6, rs7, ss8, ra9, sa10, tau11,
%           tp12, ty13, kap14, bet15, alf16, Qss17,...
%           eta18, c19, gam20, del21, nss22, nu23, sigMEpi24,
%           sigMEn25, sigMEtg26, sigMEr27, sigMEs28]

% Construct transformed initial parameter vector (thetaModeInv)
% These are the parameters to be estimated, mapped to unconstrained space if needed
thetaModeInv = [parvec(23)/(1-gam) parvec(14) parvec(16) ...
                parvec(12) parvec(13) parvec(2) parvec(4) parvec(8) ...
                parvec(6) parvec(10) parvec(1) parvec(3) parvec(7) ...
                parvec(5) parvec(9) parvec(26:30)' pissmean rssmean];

numTheta     = length(thetaModeInv);  % Total number of parameters to draw

% Set proposal covariance matrix for random walk Metropolis
proposalVar   = covscale*ParamsEstimLogLinCov;

% Preallocate storage for draws and likelihoods
thetaDraws    = NaN(numDraws,numTheta);     % Transformed parameter draws
thetaDrawsInv = NaN(numDraws,numTheta);     % Inverse-transformed (structural) draws
likVec        = NaN(numDraws,1);            % Log-posterior values

% Initialize first draw using transformed mode
thetaDraws(1,:) = transformParamsME(thetaModeInv);

% Inverse-transform back to structural space
thetaDrawInv    = transformParamsInvME((thetaDraws(1,:)));

% Evaluate posterior at initial draw
likVec(:,1)     = logl + log(modelPrior(thetaDrawInv));

% Store current values
curLik            = logl + log(modelPrior(thetaDrawInv));  % Initial log-posterior
curTheta          = thetaDraws(1,:);                        % Current transformed draw
numAcceptances    = 0;                                      % Count of accepted draws
decisionRatio     = 0;                                      % Placeholder for acceptance ratio

% Display initial log-likelihood value
fprintf("Initial likelihood %4.2f \n",logl)

% Set random seed for reproducibility of MCMC path
rng(110);

ttt = tic;  % Start timer to measure total sampling time

for draw = 1:numDraws

    % Store the current draw
    likVec(draw)           = curLik;                     % Current log posterior
    thetaDraws(draw,:)     = curTheta;                   % Current transformed parameters
    thetaDrawsInv(draw,:)  = transformParamsInvME(curTheta(:)');  % Back-transform to structural scale

    % Propose a new draw using a Random Walk Metropolis step
    newDraw     = mvnrnd(curTheta, proposalVar)';        % Multivariate normal perturbation
    newDrawInv  = transformParamsInvME(newDraw');        % Inverse-transform proposed parameters

    % Update parameter vector with proposed values
    % Structural shocks and AR(1) coefficients
    parvec(1)  = newDrawInv(11);  % rz
    parvec(2)  = newDrawInv(6);   % sz
    parvec(3)  = newDrawInv(12);  % rd
    parvec(4)  = newDrawInv(7);   % sd
    parvec(5)  = newDrawInv(14);  % rm
    parvec(6)  = newDrawInv(9);   % sm
    parvec(7)  = newDrawInv(13);  % rs
    parvec(8)  = newDrawInv(8);   % ss
    parvec(9)  = newDrawInv(15);  % ra
    parvec(10) = newDrawInv(10);  % sa

    % Fixed and structural parameters
    parvec(11) = tau;             % Fixed
    parvec(12) = newDrawInv(4);   % psi1
    parvec(13) = newDrawInv(5);   % psi2
    parvec(14) = newDrawInv(2);   % kappa
    parvec(15) = exp(newDrawInv(21) - newDrawInv(22));  % beta from transformed inflation and interest rate
    parvec(16) = newDrawInv(3);   % alpha
    parvec(17) = qss;             % Fixed
    parvec(18) = eta;             % Fixed

    % Compute derived parameters
    sigz = parvec(2) / sqrt(1 - parvec(1)^2);             % Std. dev. of z_t
    b    = (1 - gam) * newDrawInv(1) * exp(-3 * sigz);    % Surplus share b
    bet  = parvec(15);                                    % Discount factor
    c    = (1 - eta) * (1 - gam - b) * qss / (1 - bet * (1 - del));  % Vacancy cost

    parvec(19) = c;             % Vacancy cost
    parvec(20) = gam;           % Fixed
    parvec(21) = del;           % Fixed
    parvec(22) = nss;           % Fixed
    parvec(23) = b;             % Surplus share
    parvec(24) = newDrawInv(22);% Steady-state nominal rate
    parvec(25) = lambda;        % Fixed
    parvec(26:30) = newDrawInv(16:20);  % Measurement error std devs
    parvec(31) = newDrawInv(21);        % Steady-state inflation

    % Run the filter with the proposed parameters
    logl = tpfilter_zlb(parvec, Y);

    % Compute log posterior for the proposed draw
    newLik = logl + log(modelPrior(newDrawInv));

    rng(110 + 10 * draw);  % Reset random seed for reproducibility (optional)

    % Metropolis-Hastings acceptance decision
    decisionRatio = exp(newLik - curLik);
    
    if (decisionRatio > 1) || (rand < decisionRatio)
        curTheta       = newDraw;         % Accept new draw
        curLik         = newLik;
        numAcceptances = numAcceptances + 1;
        accrate        = numAcceptances / draw;
    end

    % Display progress every 200 draws
    if mod(draw, 200) == 0
        fprintf('Draw %5d | Acceptance rate: %.2f%%\n', draw, 100*accrate);
    end

end

% Report total runtime
elapst = toc(ttt);
fprintf("Time to obtain %4.0f posterior draws: %4.2f sec \n", numDraws, elapst)

%% Analyze the output and plot the priors and posteriors

% Parameter names for plotting (must match order in posterior vector)
paramnames = {"nu", "kappa", "alfa", "psi1", "psi2", ...
              "sigmaz", "sigmad", "sigmas", "sigmam", "sigmax", "rhoz", ...
              "rhod", "rhos", "rhom", "rhox", "sigPiobs", "sigNobs", ...
              "sigTGobs", "sigRobs", "sigJFobs", "piss", "rss"};

% Lower and upper bounds for x-axis limits in prior/posterior plots
% (Chosen for visual clarity with 100,000 posterior draws; may need adjustment for smaller samples)
lb = [0.5   0.0   0.5  1     0     0     0.00  0     0.000  0.000  0.85  0.8   0.4  0.8   0.4   0.000   0.0000  0.00   0      0.00   0     0.005 ];
ub = [1     0.25  0.8  5     0.4   0.01  0.03  0.06  0.01   0.02   0.98  0.95  0.8  0.99  0.95  0.005   0.002   0.05   0.003  0.05   0.02  0.03  ]; 

% Kernel density bandwidths for posterior smoothing (tuned for each parameter)
kw = [0.005  0.005  0.01  0.05  0.01  0.001  0.001  0.001  0.001  0.001 ...
      0.01   0.01   0.01  0.01  0.02  0.00005 0.00005 0.0005 0.00005 0.0005 0.0005 0.0005];

% Extract thinned and post-burn-in draws from the posterior
DrawsTKF = thetaDrawsInv(burnIn:skip:draw,:);

% Cell array of prior density functions (same order as paramnames)
priors = {@(x) nuPrior(x);         @(x) kpPrior(x);        @(x) alfaPrior(x);
          @(x) psi1Prior(x);       @(x) psi2Prior(x);       @(x) sigZPrior(x);
          @(x) sigDPrior(x);       @(x) sigSPrior(x);       @(x) sigmPrior(x);
          @(x) sigXPrior(x);       @(x) rhozPrior(x);       @(x) rhodPrior(x);
          @(x) rhoSPrior(x);       @(x) rhomPrior(x);       @(x) rhoXPrior(x);
          @(x) sigmapi_obsPrior(x);@(x) sigman_obsPrior(x); @(x) sigmatg_obsPrior(x);
          @(x) sigmar_obsPrior(x); @(x) sigmajf_obsPrior(x);@(x) pissPrior(x);
          @(x) rssPrior(x)};


% Plot prior vs posterior for each parameter
figure(5); tiledlayout('flow');  % Flexible tiling layout for subplots

for ii = 1:length(newDrawInv)
    nexttile;
    
    % Set x-grid for plotting the densities
    step = (ub(ii) - lb(ii)) / 1000;
    x = lb(ii):step:ub(ii);
    
    % Plot prior as shaded area (black)
    area(x, priors{ii}(x), 'FaceAlpha', 0.1, 'FaceColor', 'k'); 
    hold on;
    
    % Fit kernel density to posterior draws and plot (red area)
    pd = fitdist(DrawsTKF(:,ii), 'Kernel', 'Width', kw(ii)); 
    yy = pdf(pd, x); 
    area(x, yy, 'FaceAlpha', 0.3, 'FaceColor', 'r');
    
    % Format axes and title
    xlim([min(x), max(x)]);
    set(gca, 'YTickLabel', [], 'FontSize', 18, 'TickLabelInterpreter', 'latex');
    title(paramnames{ii}, 'Interpreter', 'latex'); 
    hold off;
end


%% Run the filter at the posterior mean and plot fitted observables and states

% Compute posterior mean of each parameter (after burn-in and thinning)
meanDrawsTKF = mean(DrawsTKF);

% Load posterior means into parameter vector (same ordering as estimation)
parvec(1)  = meanDrawsTKF(11);   % rz
parvec(2)  = meanDrawsTKF(6);    % sz
parvec(3)  = meanDrawsTKF(12);   % rd
parvec(4)  = meanDrawsTKF(7);    % sd
parvec(5)  = meanDrawsTKF(14);   % rm
parvec(6)  = meanDrawsTKF(9);    % sm
parvec(7)  = meanDrawsTKF(13);   % rs
parvec(8)  = meanDrawsTKF(8);    % ss
parvec(9)  = meanDrawsTKF(15);   % ra
parvec(10) = meanDrawsTKF(10);   % sa
parvec(11) = tau;                % fixed
parvec(12) = meanDrawsTKF(4);    % psi1
parvec(13) = meanDrawsTKF(5);    % psi2
parvec(14) = meanDrawsTKF(2);    % kappa
parvec(15) = exp(meanDrawsTKF(21) - meanDrawsTKF(22));  % beta from inflation - interest rate
parvec(16) = meanDrawsTKF(3);    % alpha
parvec(17) = qss;                % fixed
parvec(18) = eta;                % fixed

% Recompute derived quantities based on posterior means
sigz = parvec(2) / sqrt(1 - parvec(1)^2);                      % std. dev. of z_t
b    = (1 - gam) * meanDrawsTKF(1) * exp(-3 * sigz);           % surplus share
bet  = parvec(15);                                             % discount factor
c    = (1 - eta) * (1 - gam - b) * qss / (1 - bet * (1 - del));% vacancy cost

parvec(19) = c;
parvec(20) = gam;             % fixed
parvec(21) = del;             % fixed
parvec(22) = nss;             % fixed
parvec(23) = b;
parvec(24) = meanDrawsTKF(22);  % rss
parvec(25) = lambda;            % fixed
parvec(26:30) = meanDrawsTKF(16:20);  % measurement errors
parvec(31) = meanDrawsTKF(21);        % piss

% Run the filter at the posterior mean
[logl, a, P, yhat, omgt] = tpfilter_zlb(parvec, Y);

% Plot actual vs fitted observables
figure(2); tiledlayout(2,3);
variablenames = {"pi", "n", "tg", "r", "s"};  % Observable names
for ii = 1:5
    nexttile;
    plot(Y(:,ii)); hold on;
    plot(yhat(:,ii)); 
    title(variablenames{ii}, 'interpreter', 'latex'); 
    hold off;
end
legend({"Data", "Fitted Value"}, 'Orientation', 'horizontal', 'Location', 'southoutside')

% Plot filtered latent states (normalized)
figure(3); tiledlayout(2,3);
norm_vec = [1;
            parvec(2)/sqrt(1 - parvec(1)^2);
            parvec(4)/sqrt(1 - parvec(3)^2);
            parvec(6)/sqrt(1 - parvec(5)^2);
            parvec(8)/sqrt(1 - parvec(7)^2);
            parvec(10)/sqrt(1 - parvec(9)^2)];
statenames = {"nlag", "z", "d", "m", "s", "a"};  % State variable names
for ii = 1:6
    nexttile;
    plot(a(:,ii) ./ norm_vec(ii)); 
    title(statenames{ii}, 'interpreter', 'latex'); 
    hold off;
end
