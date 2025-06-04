function out = nkdmp_zlb_tpcoef(xt,B0,pvec)

% This function recovers the policy coefficients b of the NKDMP model
% for a given state vector xt and parameter vector pvec.
% It uses a Newton solver with Broyden updates to solve the nonlinear system 
% defined by the residuals and Jacobian from Res_eval_zlb and Jac_eval_zlb.
% The output includes the policy coefficients along with 
% the smoothed nominal interest rate function coefficients at the ZLB.

%---------------------------------------------------------------------------------------------------
% Define objects
x  = xt(:);   % States: column vector [n_{t-1}, z_t, d_t, m_t, s_t, a_t]
b0 = B0(:);   % Initial guess for policy coefficients b = [b10,..,b16, b20,...,b26, b30,...,b36, b40,...,b46]
p  = pvec(:); % Parameters: [rz, sz, rd, sd, rm, sm, rs, ss, ra, sa, tau, tp, ty, kap, bet, alf, Qss, eta, ...
              %              c, gam, del, nss, nu, rss, lam]
              % pindex:     [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  11, 12, 13,  14,  15,  16,  17,  18, ...
              %              19, 20,  21,  22, 23,  24,  25]

%---------------------------------------------------------------------------------------------------
% Newton solver setup
tol     = 1;           % initialize tolerance
maxiter = 1000;        % maximum number of iterations
crit    = 1e-6;        % convergence threshold
B       = b0;          % initialize policy coefficients
count   = 0;           % iteration counter
F       = Res_eval_zlb(x,B,p);       % initial residual vector
J       = Jac_eval_zlb(x,B,p);       % initial Jacobian matrix

while tol > crit && count <= maxiter

    % Newton step: update coefficients
    db = -J\F;         % solve linear system J*db = -F
    Bp = B + db;       % tentative updated coefficients

    % Check convergence
    count = count + 1;
    tol1  = norm(F);                           % residual norm
    tol2  = norm(abs(Bp-B)./(abs(B)+1));       % relative update norm
    tol   = max(tol1, tol2);                   % convergence criterion

    % Update coefficients and residuals
    B   = Bp;                                  % accept new guess
    F1  = Res_eval_zlb(x,B,p);                 % new residuals
    J   = J + ((F1 - F - J*db)*db')/(db'*db);  % Broyden rank-one update
    F   = F1;

    % Numerical update of Jacobian every iteration (could speed up, but adds robustness)
    if mod(count,1)==0
        J = Jac_eval_zlb(x,B,p);               % update Jacobian analytically
    end

end

%---------------------------------------------------------------------------------------------------
% Output

% Compute intercept and slope for nominal interest rate rule 
% under smoothed ZLB approximation
r   = NomR_zlb(x,Bp,p);             % level of nominal interest rate
rx  = NomR_dx_zlb(x,Bp,p);          % derivative wrt states
b0r = r - rx * x;                   % back out intercept term

% Append interest rate rule coefficients to policy vector
Bp  = [Bp; b0r; rx'];

% Return final coefficient vector
out = Bp;

end

