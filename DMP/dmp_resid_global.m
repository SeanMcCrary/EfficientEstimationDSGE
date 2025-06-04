function F = dmp_resid_global(zgrid,coef,par)

% Parameters
zdim  = length(zgrid);
p     = zdim-1;

alfa  = par.alfa; 
b     = par.b; 
bet   = par.bet; 
c     = par.c; 
del   = par.del; 
eta   = par.eta; 
rhoz  = par.rhoz; 
egrid = par.egrid;
eprob = par.ewgt';
edim  = length(egrid);

% Policies and derivatives
Tg  = zeros(zdim,1);
Tgp = zeros(zdim,edim);

for zi = 1:zdim
    Tg(zi) = sum(coef.*((zgrid(zi).^(0:p))'));

    for ei = 1:edim
        zp = zgrid(zi)*(rhoz) + egrid(ei);
        Tgp(zi,ei) = sum(coef.*((zp.^(0:p))'));
    end

end

qinv  = (1+Tg.^alfa).^(1/alfa); 
qinvp = (1+Tgp.^alfa).^(1/alfa); 

% Residuals
F = zeros(zdim,1); 
for ri =1:zdim
    F(ri) = qinv(ri) - (1-eta)*(exp(zgrid(ri))-b)/c - bet*(1-del)*sum(qinvp(ri,:).*eprob);
end
F = real(F); 

end