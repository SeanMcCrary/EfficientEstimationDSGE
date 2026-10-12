function [pi_star, Fval, it] = DPsolve_pi_NKDMP_noZLB(pi0, s, sp, b, p)

    maxit    = 30;
    tolF     = 1e-8;
    tolStep  = 1e-12;
    derivTol = 1e-12;

    pi = pi0;

    for it = 1:maxit

        % Residual and derivative at current guess
        F  = DP_NKDMP_nozlb_Res_eval(pi, s, sp, b, p);
        Fp = DP_NKDMP_nozlb_Jac_eval(pi, s, sp, b, p);

        % Guard against NaN/Inf
        if ~isfinite(F) || ~isfinite(Fp)
            warning('Non-finite residual or derivative in DP Newton solver.');
            break;
        end

        % Residual convergence
        if abs(F) < tolF
            break;
        end

        % Guard against tiny derivative
        if abs(Fp) < derivTol
            warning('Derivative too small, aborting Newton.');
            break;
        end

        % Newton step
        step = -F / Fp;
        pi   = pi + step;

        % Optional step convergence
        if abs(step) < tolStep
            break;
        end
    end

    % Final residual at returned pi
    Fval    = DP_NKDMP_nozlb_Res_eval(pi, s, sp, b, p);
    pi_star = pi;

end