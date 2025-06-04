function paramVec = transformParamsME(paramVec)
% TRANSFORMPARAMSME - Maps bounded or positive parameters to R^n (unconstrained space)
%
% This is the forward transformation (for MCMC proposals)
% - Logistic transform for (0,1)-bounded parameters
% - Log transform for strictly positive parameters

    paramVec = paramVec';  % ensure row vector input

    % Transform (0,1) variables using inverse logit
    nu    = -log(1 ./ paramVec(1,:) - 1);    % outside option 
    alfa  = -log(1 ./ paramVec(3,:) - 1);    % matching elasticity
    rhoz  = -log(1 ./ paramVec(11,:) - 1);   % persistence of z
    rhod  = -log(1 ./ paramVec(12,:) - 1);   % persistence of d
    rhos  = -log(1 ./ paramVec(13,:) - 1);   % persistence of s
    rhor  = -log(1 ./ paramVec(14,:) - 1);   % persistence of m
    rhox  = -log(1 ./ paramVec(15,:) - 1);   % persistence of a

    % Transform positive-only parameters using log
    phip         = log(paramVec(2,:));       % Price adjustment 
    psi1         = log(paramVec(4,:));       % Taylor rule inflation
    psi2         = log(paramVec(5,:));       % Taylor rule output
    sigmaz       = log(paramVec(6,:));       % std dev of z shocks
    sigmad       = log(paramVec(7,:));       % std dev of d shocks
    sigmas       = log(paramVec(8,:));       % std dev of s shocks
    sigmar       = log(paramVec(9,:));       % std dev of m shocks
    sigmax       = log(paramVec(10,:));      % std dev of a shocks
    sigmapi_obs  = log(paramVec(16,:));      % measurement error: inflation
    sigman_obs   = log(paramVec(17,:));      % measurement error: employment
    sigmatg_obs  = log(paramVec(18,:));      % measurement error: tightness
    sigmar_obs   = log(paramVec(19,:));      % measurement error: interest rate
    sigmas_obs   = log(paramVec(20,:));      % measurement error: job finding
    piss         = log(paramVec(21,:));      % steady-state inflation
    rss          = log(paramVec(22,:));      % steady-state interest rate

    % Return vector in unconstrained space (for use in MCMC)
    paramVec = [nu; phip; alfa; psi1; psi2; sigmaz; sigmad; sigmas; sigmar; ...
                sigmax; rhoz; rhod; rhos; rhor; rhox; ...
                sigmapi_obs; sigman_obs; sigmatg_obs; sigmar_obs; sigmas_obs; ...
                piss; rss];
end
