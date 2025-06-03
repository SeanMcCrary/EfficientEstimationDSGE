function out = rbcar1mc_tpcoef(xt,B0,pvec)

% This function recovers the policy coefficients b of the NKDMP model
% for a given state vector xt and set of parameters in pvec.

%---------------------------------------------------------------------------------------------------
% Define objects
x  = xt(:);   % States: x = [k_t, z_t] 
b0 = B0(:);   % Policy coeffs: b = [a00,a01,a02,b00,b01,b02,a10,a11,a12,b10,b11,b12]
p  = pvec(:); % p      = [rz0,sz0,rz1,sz1,p0,p1,tau,bet,alf,del]
              % pindex = [  1,  2,  3,  4, 5, 6,  7,  8,  9, 10] 

%---------------------------------------------------------------------------------------------------
% Newton solver initialization
maxiter = 1000;
crit    = 1e-12;
B       = b0;
F       = Res_eval(x, B, p);
J       = Jac_eval(x, B, p);
tol     = 1;
count   = 0;

%---------------------------------------------------------------------------------------------------
% Newton iteration
while tol > crit && count <= maxiter
    db  = -J \ F;
    Bp  = B + db;

    % check convergence
    count = count + 1;
    tol1  = norm(F);
    tol2  = norm(Bp - B) / (norm(B) + 1e-8);
    tol   = max(tol1, tol2);

    % update coefficients and functions
    B = Bp;
    F = Res_eval(x, B, p);
    J = Jac_eval(x, B, p);
end

if count > maxiter
    warning('Max iterations exceeded in rbcar1mc_tpcoef');
end

%---------------------------------------------------------------------------------------------------
% output
out = reshape(Bp, 6, 2)';

end

