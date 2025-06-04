function paramVec = transformParamsInvME(paramVecInv)
% TRANSFORMPARAMSINVME - Inverse transform from unconstrained to bounded parameter space
%
% INPUT:
%   paramVecInv : parameter vector in unconstrained space (e.g., R^n)
%
% OUTPUT:
%   paramVec    : transformed vector where each element respects its economic bounds
%                 - probabilities in (0,1): logistic transform
%                 - positive parameters: exponential transform

    paramVec = paramVecInv';  % row vector format

    % Bounded parameters (0,1) via logistic transform
    nu     = 1 ./ (1 + exp(-paramVec(1,:)));     % outside option 
    alfa   = 1 ./ (1 + exp(-paramVec(3,:)));     % matching elasticity
    rhoz   = 1 ./ (1 + exp(-paramVec(11,:)));    % AR(1) persistence of z
    rhod   = 1 ./ (1 + exp(-paramVec(12,:)));    % persistence of d
    rhos   = 1 ./ (1 + exp(-paramVec(13,:)));    % persistence of s
    rhor   = 1 ./ (1 + exp(-paramVec(14,:)));    % persistence of m
    rhox   = 1 ./ (1 + exp(-paramVec(15,:)));    % persistence of a

    % Positive parameters via exponential transform
    phip         = exp(paramVec(2,:));           % Price adjustment 
    psi1         = exp(paramVec(4,:));           % Taylor rule inflation
    psi2         = exp(paramVec(5,:));           % Taylor rule output
    sigmaz       = exp(paramVec(6,:));           % std dev of z shocks
    sigmad       = exp(paramVec(7,:));           % std dev of d shocks
    sigmas       = exp(paramVec(8,:));           % std dev of s shocks
    sigmar       = exp(paramVec(9,:));           % std dev of m shocks
    sigmax       = exp(paramVec(10,:));          % std dev of a shocks
    sigmapi_obs  = exp(paramVec(16,:));          % measurement error: inflation
    sigman_obs   = exp(paramVec(17,:));          % measurement error: employment
    sigmatg_obs  = exp(paramVec(18,:));          % measurement error: tightness
    sigmar_obs   = exp(paramVec(19,:));          % measurement error: interest rate
    sigmas_obs   = exp(paramVec(20,:));          % measurement error: job finding
    piss         = exp(paramVec(21,:));          % steady-state inflation
    rss          = exp(paramVec(22,:));          % steady-state interest rate

    % Repack as row vector (1 × n)
    paramVec = [nu; phip; alfa; psi1; psi2; sigmaz; sigmad; sigmas; sigmar; ...
                sigmax; rhoz; rhod; rhos; rhor; rhox; sigmapi_obs; sigman_obs; ...
                sigmatg_obs; sigmar_obs; sigmas_obs; piss; rss]';

end
