function [pi_star, Fval, it, converged] = DPsolve_pi_noZLB(pi0, s, b, p)
% Eva F. Janssens and Sean McCrary
% "Efficient Estimation of Nonlinear DSGE Models"

    maxit = 30;
    tol   = 1e-8;

    pi = pi0;
    converged = false;
    residual_is_current = false;

    for it = 1:maxit

        % Residual at current guess
        F = DP_nozlb_Res_eval(pi, s, b, p);   % F(pi_k)
        residual_is_current = true;

        if ~isreal(F) || ~isfinite(F)
            break;
        end

        if abs(F) < tol
            converged = true;
            break;
        end

        % Derivative at current guess
        Fp = DP_nozlb_Jac_eval(pi, s, b, p);   % F'(pi_k)

        % Guard against zero / tiny derivative
        if ~isreal(Fp) || ~isfinite(Fp) || abs(Fp) < 1e-6
            break;
        end

        % Newton step
        step = -F/Fp;
        pi_new = pi + step;

        if ~isreal(pi_new) || ~isfinite(pi_new)
            break;
        end

        pi = pi_new;
        residual_is_current = false;
    end

    pi_star = pi;
    if residual_is_current
        Fval = F;
    else
        Fval = DP_nozlb_Res_eval(pi_star, s, b, p);
    end
    converged = converged || ...
        (isreal(Fval) && isfinite(Fval) && abs(Fval) < tol);
    
end
