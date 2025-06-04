function paramVec = transformParamsME(paramVec)

    % Transform parameters 

    paramVec = paramVec';

    phip      = log(paramVec(1,:));         % positive
    phiy      = log(paramVec(2,:));         % positive
    phipi     = log(paramVec(3,:));         % positive
    rhom      = -log(1./paramVec(4,:)-1);   % between 0 and 1
    sigem     = log(paramVec(5,:));         % positive
    rhod      = -log(1./paramVec(6,:)-1);   % between 0 and 1
    siged     = log(paramVec(7,:));         % positive    
    rhoz      = -log(1./paramVec(8,:)-1);   % between 0 and 1
    sigez     = log(paramVec(9,:));         % positive
    tau       = log(paramVec(10,:));        % positive
    piss      = log(paramVec(11,:));        % positive
    Rss       = log(paramVec(12,:));        % positive
    sigmay_obs    = log(paramVec(13,:));    % positive
    sigmapi_obs   = log(paramVec(14,:));    % positive
    sigmar_obs    = log(paramVec(15,:));    % positive

    paramVec  = [phip; phiy; phipi; rhom; sigem; rhod; siged;...
                 rhoz; sigez; tau; piss; Rss; sigmay_obs; ...
                 sigmapi_obs; sigmar_obs];
end