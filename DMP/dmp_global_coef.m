function out = dmp_global_coef(par,deg)

alfa  = par.alfa; 
b     = par.b; 
bet   = par.bet; 
c     = par.c; 
del   = par.del; 
eta   = par.eta; 
rhoz  = par.rhoz; 
sige  = par.sige; 
qss   = (1-bet*(1-del))*c/((1-eta)*(1-b));                              % ss vacancy-filling
tgss  = (qss^(-alfa) - 1)^(1/alfa);                                        % ss market tightness
nsig  = par.nsig;
sigz  = sige/sqrt(1-rhoz^2);
zmin  = -nsig*sigz;
zmax  =  nsig*sigz;

par.zmin = zmin;
par.zmax = zmax;

tolc    = 1;
maxiter = deg;
zdim    = 2;     % number of cheb nodes, degree of cheb poly -1
count   = 2;
coef0   = [log(tgss);0];

tol     = 1e-16;
options = optimoptions('fsolve','Display','none','FunctionTolerance',tol,'OptimalityTolerance',tol,'StepTolerance',tol);

tic
while tolc >1e-8 && count <= maxiter

    zgrid = (zmin+zmax)/2 + ((zmax-zmin)/2)*cos(((2*(0:(zdim-1)) +1)/(2*zdim))*pi);

    fun   = @(coef) dmp_resid_global(zgrid,coef,par); 
    x0    = coef0;
    zcoef = real(fsolve(fun,x0,options));

    tolc  = norm(zcoef(end));
    count = count+1;
    zdim  = zdim+1;
    coef0 = [zcoef;0];

end

out = zcoef;

end