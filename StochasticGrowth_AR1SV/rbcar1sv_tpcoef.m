function out = rbcar1sv_tpcoef(xt,B0,pvec)

% This function recoveres the policy coefficients b of the NKDMP model
% for a given state vector xt and set of parameters in pvec.

%---------------------------------------------------------------------------------------------------
% Define objects
x  = xt(:);   % States: x = [k_t, z_t, s_t]
b0 = B0(:);   % Policy coeffs: b = [a0,a1,a2,a3,b0,b1,b2,b3]
p  = pvec(:); % p      = [rz,mu,rx,sx,tau,bet,alf,del,n1,n2,n3,n4,n5,w1,w2,w3,w4,w5]
              % pindex = [ 1, 2, 3, 4,  5,  6,  7,  8, 9,10,11,12,13,14,15,16,17,18]

%---------------------------------------------------------------------------------------------------
% Newton solver
tol     = 1;
maxiter = 1000;
crit    = 1e-15;
B       = b0;
count   = 0;
F       = Res_eval(x,B,p);
J       = Jac_eval(x,B,p);

while tol > crit && count<=maxiter

    % Newton solver
    db = -J\F;
    Bp = B + db;

    % check convergence
    count = count +1;
    tol1  = norm(F);
    tol2  = norm(db,'Inf');
    tol   = max(tol1,tol2);

    % update coefficients and functions
    B   = Bp;
    F   = Res_eval(x,B,p);
    J   = Jac_eval(x,B,p); % update of Jacobian

end

%---------------------------------------------------------------------------------------------------
% output
out = Bp;

end
