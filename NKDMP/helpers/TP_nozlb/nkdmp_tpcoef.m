function out = nkdmp_tpcoef(xt,B0,pvec)
% Fast plain-vanilla Newton solver for local TP coefficients.
%
% Inputs
%   xt   : state vector x_t = [hat n_t, z_t, d_t, m_t, s_t, a_t]'
%   B0   : initial guess for coefficients (28x1)
%   pvec : parameter vector (22x1)
%
% Output
%   out.B         : coefficient vector
%   out.converged : true/false
%   out.iter      : number of iterations
%   out.resnorm   : final residual norm (inf norm)
%   out.stepnorm  : final relative step norm

%--------------------------------------------------------------------------
% 0. Inputs

x = xt(:);
B = B0(:);
p = pvec(:);

if numel(x) ~= 6
    error('xt must be a 6x1 state vector.');
end
if numel(B) ~= 28
    error('B0 must be a 28x1 coefficient vector.');
end
if numel(p) ~= 22
    error('pvec must be a 22x1 parameter vector.');
end

%--------------------------------------------------------------------------
% 1. Solver options

maxiter = 50;
tolF    = 1e-12;
tolStep = 1e-8;

%--------------------------------------------------------------------------
% 2. Initial residual

F = Res_eval_tp(x,B,p);

if any(~isfinite(F))
    error('Initial residual contains NaN or Inf.');
end

resnorm   = norm(F,inf);
stepnorm  = inf;
converged = false;

if resnorm < tolF
    out.B         = B;
    out.converged = true;
    out.iter      = 0;
    out.resnorm   = resnorm;
    out.stepnorm  = 0;
    return
end

%--------------------------------------------------------------------------
% 3. Newton iterations

for iter = 1:maxiter

    J = Jac_eval_tp(x,B,p);

    if any(~isfinite(J(:)))
        warning('Jacobian contains NaN or Inf at iteration %d.',iter);
        break
    end

    % Plain Newton step
    dB = -J \ F;

    if any(~isfinite(dB))
        warning('Newton step contains NaN or Inf at iteration %d.',iter);
        break
    end

    % Relative step norm
    stepnorm = norm(dB ./ max(1,abs(B)), inf);

    % Update
    B = B + dB;

    % Residual at new iterate
    F = Res_eval_tp(x,B,p);

    if any(~isfinite(F))
        warning('Residual contains NaN or Inf after update at iteration %d.',iter);
        break
    end

    resnorm = norm(F,inf);

    % Convergence
    if resnorm < tolF
        converged = true;
        break
    end

    if stepnorm < tolStep && resnorm < 1e-6
        converged = true;
        break
    end
end

%--------------------------------------------------------------------------
% 4. Output

out.B         = B;
out.converged = converged;
out.iter      = iter;
out.resnorm   = resnorm;
out.stepnorm  = stepnorm;

if ~converged
    warning('nkdmp_tpcoef did not converge. iter=%d, resnorm=%g, stepnorm=%g', ...
        iter,resnorm,stepnorm);
end

end