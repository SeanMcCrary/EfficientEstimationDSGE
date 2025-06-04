clc
clear
close all;
rng(1995);

% load the data and the proposal covariance matrix 
load("demo_data.mat")

% this scale is used to get an acceptance rate around 30% after burnin
covscale = 0.25;

% how many of the draws are burnin, how many in total?
numDraws     = 100000;
burnIn       = 20000;
skip         = 5;      % if you want to do trimming at the end

% set initial values for all parameters (some are fixed)
gam   = 0.1;                  % ss markdown (monopolistic competition) 
zss   = 1;                    % ss tfp (level) 
yss   = 1;                    % ss output 
phip  = 100;                  % rotemberg price parameter
wss   = (1-gam)*zss;          % ss wage 
chi   = wss;                  % labor disutility(consistent with lss=1); 
lss   = 1;                    % ss labor 
piss  = 0.0154;               % steady state inflation 
Rss   = 0.0582;               % ss nominal rate 
beta  = exp(piss-Rss);        % discount factor 
phipi = 2;                    % taylor rule - inflation 
phiy  = 0.5;                  % taylor rule - output 
tau   = 2;                    % ies - 1=log utility 
eta   = 5;                    % inverse frisch 

% exogenous processes
rhoz  = 0.8972;                % tfp - persistence 
sigez = 0.0039;                % tfp - innovation 
sigz  = sigez/sqrt(1-rhoz^2);
rhom  = 0.9;                   % monetary - persistence 
sigem = 0.0101;                % monetary - innovation 
sigm  = sigem/sqrt(1-rhom^2);
rhod  = 0.9206;                % discount - persistence 
siged = 0.1430;                % discount - innovation 
sigd  = siged/sqrt(1-rhod^2);

% collect parameters to be passed to TP 
par.gamma         = gam;
par.zss           = zss; 
par.yss           = yss; 
par.phip          = phip; 
par.wss           = wss; 
par.chi           = chi; 
par.lss           = lss;
par.piss          = piss; 
par.bet           = beta; 
par.phipi         = phipi; 
par.phiy          = phiy;
par.Rss           = Rss; 
par.tau           = tau; 
par.eta           = eta; 
par.rhoz          = rhoz; 
par.sigez         = sigez; 
par.rhod          = rhod; 
par.siged         = siged; 
par.rhom          = rhom; 
par.sigem         = sigem; 

% stack observables
Y = [y_obs pi_obs r_obs];

% initialize measurement errors
par.sigMEy        = 0.07*std(y_obs);
par.sigMEpi       = 0.07*std(pi_obs);
par.sigMEr        = 0.07*std(r_obs);

% do a first run of the TP filter
[logl,~,~,~] = tpfilter(par,Y);

% make all your priors:

phipMean    = 50; phipStd = 10;
phipScale   = (phipStd^2)/phipMean;
phipShape   = phipMean/phipScale;
truncU      = 200;
truncL      = 0;
phipPrior   = @(phip) (phip<=200).*gampdf(phip,phipShape,phipScale)./(gamcdf(truncU,phipShape,phipScale)-gamcdf(truncL,phipShape,phipScale));

phiyMean    = 0.75; phiyStd = 0.25;
phiyScale   = (phiyStd^2)/phiyMean;
phiyShape   = phiyMean/phiyScale;
phiyPrior   = @(phiy) gampdf(phiy,phiyShape,phiyScale);

phipiMean    = 2; phipiStd = 0.25;
phipiScale   = (phipiStd^2)/phipiMean;
phipiShape   = phipiMean/phipiScale;
truncL       = 1;
phipiPrior   = @(phipi) (phipi>=truncL).*(gampdf(phipi,phipiShape,phipiScale)./(1-gamcdf(truncL,phipiShape,phipiScale)));

rhomMean    = 0.7; rhomStd = 0.15; 
rhomA       = rhomMean*(rhomMean*(1-rhomMean)/(rhomStd^2)-1);
rhomB       = (1-rhomMean)*(rhomMean*(1-rhomMean)/(rhomStd^2)-1);
rhomPrior   = @(rhom) betapdf(rhom,rhomA,rhomB);

sigMmean    = 0.005; sigMStd = 0.01;
sigMShockA  = (sigMmean^2)/(sigMStd^2) + 2;
sigMShockB  = sigMmean*((sigMmean^2)/(sigMStd^2) + 1);
sigMPrior   = @(x) (sigMShockB^sigMShockA)/gamma(sigMShockA) * x.^(-sigMShockA-1) .* exp(-sigMShockB ./ x);

rhodMean    = 0.7; rhodStd = 0.15; 
rhodA       = rhodMean*(rhodMean*(1-rhodMean)/(rhodStd^2)-1);
rhodB       = (1-rhodMean)*(rhodMean*(1-rhodMean)/(rhodStd^2)-1);
rhodPrior   = @(rhod) betapdf(rhod,rhodA,rhodB);

sigDmean    = 0.05; sigDStd = 0.05;
sigDShockA  = (sigDmean^2)/(sigDStd^2) + 2;
sigDShockB  = sigDmean*((sigDmean^2)/(sigDStd^2) + 1);
sigDPrior   = @(x) (sigDShockB^sigDShockA)/gamma(sigDShockA) * x.^(-sigDShockA-1) .* exp(-sigDShockB ./ x);

rhozMean    = 0.7; rhozStd = 0.15; 
rhozA       = rhozMean*(rhozMean*(1-rhozMean)/(rhozStd^2)-1);
rhozB       = (1-rhozMean)*(rhozMean*(1-rhozMean)/(rhozStd^2)-1);
rhozPrior   = @(rhoz) betapdf(rhoz,rhozA,rhozB);

sigZmean    = 0.01; sigZStd = 0.01;
sigZShockA  = (sigZmean^2)/(sigZStd^2) + 2;
sigZShockB  = sigZmean*((sigZmean^2)/(sigZStd^2) + 1);
sigZPrior   = @(x) (sigZShockB^sigZShockA)/gamma(sigZShockA)*x.^(-sigZShockA-1).*exp(-sigZShockB./x);

tauMean    = 2; tauStd = 0.5;
tauScale   = (tauStd^2)/tauMean;
tauShape   = tauMean/tauScale;
truncL     = 1;
tauPrior   = @(x) (x>=truncL).*gampdf(x,tauShape,tauScale)/(1-gamcdf(truncL,tauShape,tauScale));

pissMean    = 0.01; pissStd = 0.005;
pissScale   = (pissStd^2)/pissMean;
pissShape   = pissMean/pissScale;
pissPrior   = @(x) gampdf(x,pissShape,pissScale);

rssMean    = 0.016; rssStd = 0.005;
rssScale   = (rssStd^2)/rssMean;
rssShape   = rssMean/rssScale;
rssPrior   = @(x) gampdf(x,rssShape,rssScale);

sigmay_obsMean    = 0.005; sigmay_obsStd = 2; 
sigmay_obsShockA  = (sigmay_obsMean^2)/(sigmay_obsStd^2) + 2;
sigmay_obsShockB  = sigmay_obsMean * ((sigmay_obsMean^2)/(sigmay_obsStd^2) + 1);
sigmay_obsPrior   = @(x) (sigmay_obsShockB^sigmay_obsShockA)/gamma(sigmay_obsShockA) * x.^(-sigmay_obsShockA-1) .* exp(-sigmay_obsShockB ./ x);

sigmapi_obsMean    = 0.005; sigmapi_obsStd = 2;
sigmapi_obsShockA  = (sigmapi_obsMean^2)/(sigmapi_obsStd^2) + 2;
sigmapi_obsShockB  = sigmapi_obsMean * ((sigmapi_obsMean^2)/(sigmapi_obsStd^2) + 1);
sigmapi_obsPrior   = @(x) (sigmapi_obsShockB^sigmapi_obsShockA)/gamma(sigmapi_obsShockA) * x.^(-sigmapi_obsShockA-1) .* exp(-sigmapi_obsShockB ./ x);

sigmar_obsMean    = 0.005; sigmar_obsStd = 2; 
sigmar_obsShockA  = (sigmar_obsMean^2)/(sigmar_obsStd^2) + 2;
sigmar_obsShockB  = sigmar_obsMean * ((sigmar_obsMean^2)/(sigmar_obsStd^2) + 1);
sigmar_obsPrior   = @(x) (sigmar_obsShockB^sigmar_obsShockA)/gamma(sigmar_obsShockA) * x.^(-sigmar_obsShockA-1) .* exp(-sigmar_obsShockB ./ x);

% Collect all priors together:
modelPrior = @(x) phipPrior(x(1)).*phiyPrior(x(2)).*phipiPrior(x(3))...
                  .*rhomPrior(x(4)).*sigMPrior(x(5)).*rhodPrior(x(6)).*...
                    sigDPrior(x(7)).*rhozPrior(x(8)).*sigZPrior(x(9)).*...
                    tauPrior(x(10)).*pissPrior(x(11)).*rssPrior(x(12)).*...
                    sigmay_obsPrior(x(13)).*sigmapi_obsPrior(x(14)).*sigmar_obsPrior(x(15));

% collect initial parameters
thetaModeInv = [par.phip par.phiy par.phipi par.rhom par.sigem par.rhod...
                par.siged par.rhoz par.sigez par.tau par.piss par.Rss ...
                par.sigMEy par.sigMEpi par.sigMEr];

numTheta     = length(thetaModeInv); % how many parameters?

% Set your proposal variance:
proposalVar = covscale*ParamsEstimLogLinCov;

thetaDraws    = NaN(numDraws,numTheta);
thetaDrawsInv = NaN(numDraws,numTheta);
likVec        = NaN(numDraws,1);

% collect the initial parameters and posterior value
thetaDraws(1,:) = transformParamsME(thetaModeInv);
thetaDrawInv    = transformParamsInvME((thetaDraws(1,:)'));
likVec(1)       = logl + log(modelPrior(thetaDrawInv));

curLik0            = logl + log(modelPrior(thetaDrawInv));
curTheta0          = thetaDraws(1,:);
numAcceptances     = 0;
decisionRatio      = 0;  
curLik             = curLik0;
curTheta           = curTheta0;

tic

for draw = 1:numDraws
    % set seed for current draw
    rng(110+10*draw);

    % store the last draws 
    likVec(draw)          = curLik;
    thetaDraws(draw,:)    = curTheta;
    thetaDrawsInv(draw,:)    = transformParamsInvME(curTheta(:));

    % new draw from the Random Walk sampler
    newDraw     = mvnrnd(curTheta,proposalVar)';
    % transform to the relevant parameter range
    newDrawInv  = transformParamsInvME(newDraw(:));

    % collect the new draws in par-structure to pass it to the TP filter
    par.phip         = newDrawInv(1);
    par.phiy         = newDrawInv(2);
    par.phipi        = newDrawInv(3);
    par.rhom         = newDrawInv(4);
    par.sigem        = newDrawInv(5);
    par.rhod         = newDrawInv(6);
    par.siged        = newDrawInv(7);
    par.rhoz         = newDrawInv(8);
    par.sigez        = newDrawInv(9);
    par.tau          = newDrawInv(10);
    par.piss         = newDrawInv(11);
    par.Rss          = newDrawInv(12);
    par.sigMEy       = newDrawInv(13);
    par.sigMEpi      = newDrawInv(14);
    par.sigMEr       = newDrawInv(15);

    % run the filter
    logl = tpfilter(par,Y);

    % posterior /propto logl + log prior
    newLik              = logl + log(modelPrior(newDrawInv));

    % decide whether to accept draw
    decisionRatio = exp(newLik-curLik);

    if (decisionRatio > 1) || (rand < decisionRatio)
            curTheta        = newDraw;
            curLik          = newLik;
            numAcceptances  = numAcceptances + 1;
    end

    % print some statistics
    if mod(draw,100) == 0
        disp(draw)
        disp(numAcceptances/draw);
    end
end

runtimet = toc;
fprintf("Time to obtain %4.0f posterior draws: %4.2f sec \n", numDraws, runtimet)

%% Plot all priors and posteriors

DrawsTKF     = thetaDrawsInv(burnIn:skip:numDraws,:);

figure(21); tiledlayout('flow'); nexttile;
x = 0:0.01:150;
area(x,phipPrior(x),'FaceAlpha',0.1,'FaceColor','k'); hold on;
title('$\phi_p$','interpreter','latex'); hold on; 
set(gca,"FontSize",18,'TickLabelInterpreter', 'latex');
pd = fitdist(DrawsTKF(1:end,1),'Kernel','Width',4); 
yy = pdf(pd,x); area(x,yy,'FaceAlpha',0.3,'FaceColor','r'); hold on;
xlim([20 140])
set(gca,'YTickLabel',[]);
hold off;

nexttile;
x = 0:0.01:0.5;
area(x,phiyPrior(x),'FaceAlpha',0.1,'FaceColor','k'); hold on;
title('$\psi_y$','interpreter','latex'); hold on; 
set(gca,"FontSize",18,'TickLabelInterpreter', 'latex');
pd = fitdist(DrawsTKF(1:end,2),'Kernel','Width',0.02); 
yy = pdf(pd,x); area(x,yy,'FaceAlpha',0.3,'FaceColor','r');
set(gca,'YTickLabel',[]);
xlim([0 0.5])
hold off;

nexttile;
x = 0.5:0.001:4;
area(x,phipiPrior(x),'FaceAlpha',0.1,'FaceColor','k'); hold on;
title('$\psi_\pi$','interpreter','latex'); hold on; 
set(gca,"FontSize",18,'TickLabelInterpreter', 'latex');
pd = fitdist(DrawsTKF(1:end,3),'Kernel','Width',0.07); 
yy = pdf(pd,x); area(x,yy,'FaceAlpha',0.3,'FaceColor','r');
set(gca,'YTickLabel',[]);
xlim([0.5 4])
hold off;

nexttile;
x = 0.5:0.001:1;
area(x,rhomPrior(x),'FaceAlpha',0.1,'FaceColor','k'); hold on;
title('$\rho_m$','interpreter','latex'); hold on; 
set(gca,"FontSize",18,'TickLabelInterpreter', 'latex');
pd = fitdist(DrawsTKF(1:end,4),'Kernel','Width',0.01); 
yy = pdf(pd,x); area(x,yy,'FaceAlpha',0.3,'FaceColor','r');
set(gca,'YTickLabel',[]);
xlim([0.5 1])
hold off;

nexttile;
x = 0:0.0001:0.015;
area(x,sigMPrior(x),'FaceAlpha',0.1,'FaceColor','k'); 
title('$\sigma_m$','interpreter','latex'); hold on; set(gca,"FontSize",18,'TickLabelInterpreter', 'latex');
pd = fitdist(DrawsTKF(1:end,5),'Kernel','Width',0.0001); 
yy = pdf(pd,x); area(x,yy,'FaceAlpha',0.3,'FaceColor','r');
set(gca,'YTickLabel',[]);
xlim([0 0.012])
hold off;

nexttile;
x = 0.85:0.001:1;
area(x,rhodPrior(x),'FaceAlpha',0.1,'FaceColor','k'); 
title('$\rho_d$','interpreter','latex'); hold on; set(gca,"FontSize",18,'TickLabelInterpreter', 'latex');
pd = fitdist(DrawsTKF(1:end,6),'Kernel','Width',0.005); 
yy = pdf(pd,x); area(x,yy,'FaceAlpha',0.3,'FaceColor','r');
set(gca,'YTickLabel',[]);
xlim([0.9 1])
hold off;

nexttile;
x = 0:0.0001:0.3;
area(x,sigDPrior(x),'FaceAlpha',0.1,'FaceColor','k'); 
title('$\sigma_d$','interpreter','latex'); hold on; 
set(gca,"FontSize",18,'TickLabelInterpreter', 'latex');
pd = fitdist(DrawsTKF(1:end,7),'Kernel','Width',0.005); 
yy = pdf(pd,x); area(x,yy,'FaceAlpha',0.3,'FaceColor','r');
set(gca,'YTickLabel',[]);
xlim([0 0.07])
hold off;

nexttile;
x = 0.8:0.001:1;
area(x,rhozPrior(x),'FaceAlpha',0.1,'FaceColor','k'); 
title('$\rho_a$','interpreter','latex'); hold on; 
set(gca,"FontSize",18,'TickLabelInterpreter', 'latex');
pd = fitdist(DrawsTKF(1:end,8),'Kernel','Width',0.003); 
yy = pdf(pd,x); area(x,yy,'FaceAlpha',0.3,'FaceColor','r');
set(gca,'YTickLabel',[]);
xlim([0.9 1])
hold off;

nexttile;
x = 0:0.00001:0.01;
area(x,sigZPrior(x),'FaceAlpha',0.1,'FaceColor','k'); 
title('$\sigma_a$','interpreter','latex'); hold on; 
set(gca,"FontSize",18,'TickLabelInterpreter', 'latex');
pd = fitdist(DrawsTKF(1:end,9),'Kernel','Width',0.0001); 
yy = pdf(pd,x); area(x,yy,'FaceAlpha',0.3,'FaceColor','r');
set(gca,'YTickLabel',[]);
xlim([0.0015 0.0045])
hold off;

nexttile;
x = 0:0.01:5;
area(x, tauPrior(x),'FaceAlpha',0.1,'FaceColor','k'); 
title('$\tau$','interpreter','latex'); hold on; 
set(gca,"FontSize",18,'TickLabelInterpreter', 'latex');
pd = fitdist(DrawsTKF(1:end,10),'Kernel','Width',0.05); 
yy = pdf(pd,x); area(x,yy,'FaceAlpha',0.3,'FaceColor','r');
set(gca,'YTickLabel',[]);
xlim([0.5 3.5])
hold off;

nexttile;
x = 0:0.0001:0.03; 
area(x,pissPrior(x),'FaceAlpha',0.1,'FaceColor','k'); 
title('$\pi_{ss}$','interpreter','latex'); hold on; 
set(gca,"FontSize",18,'TickLabelInterpreter', 'latex');
pd = fitdist(DrawsTKF(1:end,11),'Kernel','Width',0.001); 
yy = pdf(pd,x); area(x,yy,'FaceAlpha',0.3,'FaceColor','r');
set(gca,'YTickLabel',[]);
xlim([0.005 0.015])
hold off;
legend({"Prior","Nonlinear"},'orientation','horizontal','location','southoutside','box','off','interpreter','latex')

nexttile;
x = 0:0.0001:0.1;
area(x,rssPrior(x),'FaceAlpha',0.1,'FaceColor','k'); 
title('$r_{ss}$','interpreter','latex'); hold on; set(gca,"FontSize",18,'TickLabelInterpreter', 'latex');
pd = fitdist(DrawsTKF(1:end,12),'Kernel','Width',0.001); 
yy = pdf(pd,x); area(x,yy,'FaceAlpha',0.3,'FaceColor','r');
set(gca,'YTickLabel',[]);
xlim([0.005 0.03])
hold off;

%% Run filter at the posterior mean
parnonlin     = par;
AveNonLinDraw = mean(DrawsTKF);

parnonlin.phip         = AveNonLinDraw(1);
parnonlin.phiy         = AveNonLinDraw(2);
parnonlin.phipi        = AveNonLinDraw(3);
parnonlin.rhom         = AveNonLinDraw(4);
parnonlin.sigem        = AveNonLinDraw(5);
parnonlin.rhod         = AveNonLinDraw(6);
parnonlin.siged        = AveNonLinDraw(7);
parnonlin.rhoz         = AveNonLinDraw(8);
parnonlin.sigez        = AveNonLinDraw(9);
parnonlin.tau          = AveNonLinDraw(10);
parnonlin.piss         = AveNonLinDraw(11);
parnonlin.Rss          = AveNonLinDraw(12);
parnonlin.sigMEy       = AveNonLinDraw(13);
parnonlin.sigMEpi      = AveNonLinDraw(14);
parnonlin.sigMEr       = AveNonLinDraw(15);

[logl,a,P,yhat] = tpfilter(parnonlin,Y);

% Plot filter output
years = 1966:0.25:2007.75;

figure(12); tiledlayout(3,1);
nexttile;
plot(years,a(:,1),'-','LineWidth',1.5); hold on;
title("TFP",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','fontsize',16)
nexttile;
plot(years,a(:,2),'-','LineWidth',1.5); hold on;
title("DF",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','fontsize',16)
nexttile;
plot(years,a(:,3),'-','LineWidth',1.5); hold on;
title("MP",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','fontsize',16)
legend({"Filtered shock"},'interpreter','latex','fontsize',16,...
    'orientation','horizontal','box','off','location','southoutside')

figure(11); tiledlayout(3,1);
nexttile;
plot(years,yhat(:,1),'*-','LineWidth',1.5); hold on;
plot(years,Y(:,1),'LineWidth',1.5); hold off;
title("Output",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','fontsize',16)
nexttile;
plot(years,yhat(:,2),'*-','LineWidth',1.5); hold on;
plot(years,Y(:,2),'LineWidth',1.5); hold off;
title("Inflation",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','fontsize',16)
nexttile;
plot(years,yhat(:,3),'*-','LineWidth',1.5); hold on;
plot(years,Y(:,3),'LineWidth',1.5); hold off;
title("Interest rate",'interpreter','latex','fontsize',16)
set(gca,'TickLabelInterpreter','latex','fontsize',16)
legend({"Fitted value","Data"},'interpreter','latex','fontsize',16,...
    'orientation','horizontal','box','off','location','southoutside')


