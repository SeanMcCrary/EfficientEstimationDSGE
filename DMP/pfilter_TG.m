function [logl,a,yhat] = pfilter_TG(parvec,Y,M)

% Filter using the Global Solution
% and the Particle Filter
% Date: 4 feb 2025
% Authors: Eva Janssens & Sean McCrary
% Input: parvec = [alfa,b,beta,c,delta,eta,rhoz,sige,sigtg,sigw]
%        Y
% Output: logl is loglikelihood, 
%         a filtered state, 
%         yhat is filtered observables

rng(1);

Tsim = length(Y);

% Read out parameter vector:
alfa  = parvec(1);
b     = parvec(2);
beta  = parvec(3);
c     = parvec(4);
delta = parvec(5);
eta   = parvec(6);
rhoz  = parvec(7);
sige  = parvec(8);
sigtg = parvec(9);

sigz = sige/sqrt(1-rhoz^2); % log tfp sd 

[~,epsi_nodes,weight_nodes] = Monomials_2(1,sige^2);

par.c     = c; 
par.bet   = beta; 
par.alfa  = alfa; 
par.del   = delta;
par.rhoz  = rhoz; 
par.sige  = sige; 
par.eta   = eta; 
par.b     = b;
par.sigz  = sigz; 
par.egrid = epsi_nodes; 
par.ewgt  = weight_nodes; 

% Solve the model
par.nsig    = 4;
poly_degree = 9; % degree of polynomial approximation (5 is sufficient) 
dmp_global_coefs = dmp_global_coef(par,poly_degree); 

% Initialize:
atm1 = sigz*randn(M,1);
Wt   = ones(M,1);
pfilter_y = []; pfilter_a = [];

glob_t = sim_global_dmp(dmp_global_coefs,b,eta,rhoz,sige,1,atm1); 
tgt    = glob_t.tightness; 

pfilter_y(1,:) = [mean(tgt.*Wt')]; 
pfilter_a(1,:)   = mean(atm1.*Wt);
Ht   = sigtg^2; 

pfilter_logl = zeros(Tsim,1);

for it=2:Tsim

    at      = rhoz*atm1 + sige*randn(M,1);

    glob_t  = sim_global_dmp(dmp_global_coefs,b,eta,rhoz,sige,1,at); 
    tgt     = glob_t.tightness; 

    yt = Y(it,:);
    
    wt = mvnpdf(tgt', yt, Ht);
    
    Wttilde          = wt.*Wt ./  mean(wt.*Wt);
    pfilter_logl(it) = log(mean(wt.*Wt));

    ESS = M/(mean(Wttilde.^2));

    % particle selection:
    if ESS < M*0.5
        idx  = randsample(length(Wttilde), M, true, Wttilde);
        atm1 = at(idx); tgt = tgt(idx);
    else
        Wt = Wttilde; atm1 = at;
    end

    pfilter_y(it,:) = [mean(tgt.*Wt')]; 
    pfilter_a(it)   = mean(at.*Wt);

end

logl = sum(pfilter_logl);
a    = pfilter_a;
yhat = pfilter_y;

end