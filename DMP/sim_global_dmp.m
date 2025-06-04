function out = sim_global_dmp(zcoef,b,eta,rhoz,sige,T,z0)

% Simulate time series of z_t
k = length(z0);

zt      = zeros(T,k);
zt(1,:) = z0;
et      = normrnd(0,sige,[T k]);

for ti=2:T
    zt(ti,:) = rhoz*zt(ti-1,:) + et(ti,:);
end


% time series of tg_t 
p   = length(zcoef)-1; 

tgt = zeros(T,k); 
for ki=1:k
    tgt(:,ki) = ((zcoef')*((zt(:,ki).^(0:p))'))';
end 


% time series of wages 
wt = eta*exp(zt) + (1-eta)*b; 

% output 
out.tightness = tgt; 
out.wage      = wt; 
out.logtfp    = zt; 

end 