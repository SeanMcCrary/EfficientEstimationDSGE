function [q,qz,qzz] = DP_DMP(z,par,Ndp,order)
% (c) Eva F. Janssens and Sean McCrary
% These codes are part of the demo codes for the paper "Efficient Estimation 
% of Nonlinear DSGE Models" by Janssens & McCrary.

% First- or second-order Dynamic Perturbation evaluated at z.

if nargin < 4
    order = 1;
end

zsize = size(z);
z = z(:);
terminal_z = z*par.rho^(Ndp+1);

if order == 1
    q_next   = par.bss_q*terminal_z;
    qz_next  = par.bss_q*ones(size(z));
    qzz_next = zeros(size(z));

    for ii = Ndp:-1:0
        zi = z*par.rho^ii;
        M = exp(-q_next+0.5*par.sigma^2*qz_next.^2);
        D = par.kappa*(exp(zi)-par.nu)+par.gamma*M;

        q_next  = -log(D);
        qz_next = (par.gamma*par.rho*qz_next.*M ...
            -par.kappa*exp(zi))./D;
    end
else
    terminal_solution = PERT_DMP(par,2);
    [q_next,qz_next,qzz_next] = ...
        EVAL_PERT_DMP(terminal_solution,terminal_z,2);

    for ii = Ndp:-1:0
        zi = z*par.rho^ii;
        S = 1+par.sigma^2*qzz_next;
        M = exp(-q_next+0.5*par.sigma^2*qz_next.^2./S)./sqrt(S);
        Ez = exp(zi);
        D = par.kappa*(Ez-par.nu)+par.gamma*M;

        Dz = par.kappa*Ez ...
            -par.gamma*par.rho*(qz_next./S).*M;
        Dzz = par.kappa*Ez+par.gamma*par.rho^2 ...
            .*(qz_next.^2./S.^2-qzz_next./S).*M;

        q_next   = -log(D);
        qz_next  = -Dz./D;
        qzz_next = qz_next.^2-Dzz./D;
    end
end

q   = reshape(q_next,zsize);
qz  = reshape(qz_next,zsize);
qzz = reshape(qzz_next,zsize);
end
