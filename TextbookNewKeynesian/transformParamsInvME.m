function paramVec = transformParamsInvME(paramVecInv)

    % paramVec  = [phip; phiy; phipi; rhom; sigem; rhod; siged;...
    %              rhoz; sigez; tau; piss; Rss; sigmay_obs; ...
    %              sigmay_obs; sigmapi_obs; sigmar_obs];

    % Transform parameters (paramVecInv can take any value, paramVec respects its bounds)
    paramVec = paramVecInv;
    
    phip      = exp(paramVec(1,:));               % positive
    phiy      = exp(paramVec(2,:));               % positive
    phipi     = exp(paramVec(3,:));               % positive
    rhom      = 1 ./ (1 + exp(-paramVec(4,:)));   % between 0 and 1
    sigem     = exp(paramVec(5,:));               % positive
    rhod      = 1 ./ (1 + exp(-paramVec(6,:)));   % between 0 and 1
    siged     = exp(paramVec(7,:));               % positive
    rhoz      = 1 ./ (1 + exp(-paramVec(8,:)));   % between 0 and 1
    sigez     = exp(paramVec(9,:));               % positive
    tau       = exp(paramVec(10,:));              % positive
    piss      = exp(paramVec(11,:));              % positive
    Rss       = exp(paramVec(12,:));              % positive
    sigmay_obs    = exp(paramVec(13,:));          % positive
    sigmapi_obs   = exp(paramVec(14,:));          % positive
    sigmar_obs    = exp(paramVec(15,:));          % positive
   
    % collect them back together
    paramVec  = [phip; phiy; phipi; rhom; sigem; rhod; siged;...
                 rhoz; sigez; tau; piss; Rss; sigmay_obs; ...
                 sigmapi_obs; sigmar_obs]';
end