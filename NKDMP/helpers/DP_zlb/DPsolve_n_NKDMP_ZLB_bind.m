function [n_star, Fval, it, converged] = DPsolve_n_NKDMP_ZLB_bind(n0, s, sp, b, p)

    maxit        = 30;
    maxBacktrack = 5;

    tolF      = 1e-7;
    tolStep   = 1e-7;
    derivTol  = 1e-12;
    minLambda = 2^(-maxBacktrack);

    n = n0;
    converged = false;

    % Initial residual
    F = double(DP_NKDMP_zlb_bindN_Res_eval(n, s, sp, b, p));

    if ~isfinite(F) || ~isreal(F)
        warning('Initial residual is non-finite or complex in DP ZLB binding solver.');
        n_star = n;
        Fval   = F;
        it     = 0;
        return
    end

    for it = 1:maxit

        % Residual convergence
        if abs(F) < tolF
            converged = true;
            break
        end

        % Derivative at current iterate
        Fp = double(DP_NKDMP_zlb_bindN_Jac_eval(n, s, sp, b, p));

        if ~isfinite(Fp) || ~isreal(Fp)
            warning('Non-finite or complex derivative in DP ZLB binding Newton solver.');
            break
        end

        if abs(Fp) < derivTol
            warning('Derivative too small in DP ZLB binding Newton solver.');
            break
        end

        % Full Newton step
        step = -F / Fp;

        % Guarded backtracking
        lambda   = 1.0;
        accepted = false;

        for bt = 1:(maxBacktrack + 1)

            n_trial = n + lambda * step;
            F_trial = double(DP_NKDMP_zlb_bindN_Res_eval(n_trial, s, sp, b, p));

            if isfinite(F_trial) && isreal(F_trial) && (abs(F_trial) < abs(F))
                accepted = true;
                break
            end

            lambda = 0.5 * lambda;

            if lambda < minLambda
                break
            end
        end

        if ~accepted
            warning('No acceptable damped Newton step found in DP ZLB binding solver.');
            break
        end

        % Accept step
        n = n_trial;
        F = F_trial;

        if abs(F) < tolF
            converged = true;
            break
        end

        if abs(lambda * step) < tolStep
            break
        end
    end

    n_star = n;
    Fval   = F;

    if isfinite(Fval) && isreal(Fval) && abs(Fval) < tolF
        converged = true;
    end
end