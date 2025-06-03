%-------------------------------------------------------------------------
% Description 
% This file constructs the residual functions of the stochastic growth
% model with markov switching taking three vectors as arguments 
% 1. p - vector of model parameters 
% 2. x - vector of states 
% 3. b - vector of policy function coefficients 
%
% The file also constructs the analytic Jacobian as a function of these
% arguments. Both the residual function and Jacobian are defined using
% symbolic function then converted to Matlab functions that are called in
% the solution step 
%
% NOTE! this file must be run once before the solution/estimation routine 
% to store the residual and jacobian functions which are called later 
%-------------------------------------------------------------------------

clear; close; clc; 
%-------------------------------------------------------------------------
% 0. Define Arguments 

% states 
syms x [2 1] real  % x = [k_t, z_t] 

% policy coefficients 
syms b [12 1] real % b = [a00,a01,a02,b00,b01,b02,a10,a11,a12,b10,b11,b12]

% parameters 
syms p [10 1] real % p      = [rz0,sz0,rz1,sz1,p0,p1,tau,bet,alf,del]
                   % pindex = [  1,  2,  3,  4, 5, 6,  7,  8,  9, 10] 

%-------------------------------------------------------------------------
% 0. Policies & Auxiliary Equations 

% parameters 
rz0 = p(1); sz0 = p(2); rz1 = p(3); sz1 = p(4);  % AR(1)
p0  = p(5); p1  = p(6);                          % Markov 
tau = p(7); bet = p(8); alf = p(9); del = p(10); % preferences & technology 

% states z_t and k_t 
k = x(1); z = x(2); 

% policy coefficient (mapping to notes) 
a00 = b(1);  a01 = b(2);  a02 = b(3); 
b00 = b(4);  b01 = b(5);  b02 = b(6); 
a10 = b(7);  a11 = b(8);  a12 = b(9);
b10 = b(10); b11 = b(11); b12 = b(12);

% policies (consumption c_t and capital k_t+1 for each s_t in {0,1} ) 
c0  = a00 + a01*k + a02*z; 
c1  = a10 + a11*k + a12*z;
kp0 = b00 + b01*k + b02*z;
kp1 = b10 + b11*k + b12*z;

% expectation terms for Euler equations 
g00 = -tau*(a00+a01*b00+a01*b01*k+(a01*b02+a02*rz0)*z)+rz0*z+(alf-1)*kp0+(((1-tau*a02)^2)*sz0^2)/2; 
g01 = -tau*(a10+a11*b00+a11*b01*k+(a11*b02+a12*rz1)*z)+rz1*z+(alf-1)*kp0+(((1-tau*a12)^2)*sz1^2)/2; 
g10 = -tau*(a00+a01*b10+a01*b11*k+(a01*b12+a02*rz0)*z)+rz0*z+(alf-1)*kp1+(((1-tau*a02)^2)*sz0^2)/2; 
g11 = -tau*(a10+a11*b10+a11*b11*k+(a11*b12+a12*rz1)*z)+rz1*z+(alf-1)*kp1+(((1-tau*a12)^2)*sz1^2)/2; 

h00 = -tau*(a00+a01*b00+a01*b01*k+(a01*b02+a02*rz0)*z)+(((tau*a02)^2)*sz0^2)/2; 
h01 = -tau*(a10+a11*b00+a11*b01*k+(a11*b02+a12*rz1)*z)+(((tau*a12)^2)*sz1^2)/2; 
h10 = -tau*(a00+a01*b10+a01*b11*k+(a01*b12+a02*rz0)*z)+(((tau*a02)^2)*sz0^2)/2; 
h11 = -tau*(a10+a11*b10+a11*b11*k+(a11*b12+a12*rz1)*z)+(((tau*a12)^2)*sz1^2)/2; 

%-------------------------------------------------------------------------
% 1. Euler Equations (one for each initial s_t in {0,1} ) 

EE0 = -exp(-tau*c0) + ...
      (1-bet*(1-del))*(p0*exp(g00)+(1-p0)*exp(g01)) + ...
      bet*(1-del)*(p0*exp(h00)+(1-p0)*exp(h01));

EE1 = -exp(-tau*c1) + ...
      (1-bet*(1-del))*(p1*exp(g11)+(1-p1)*exp(g10)) + ...
      bet*(1-del)*(p1*exp(h11)+(1-p1)*exp(h10)); 

% residuals 
EE0x = jacobian(EE0,x); 
EE1x = jacobian(EE1,x); 

% %-------------------------------------------------------------------------
% % 2. Resource constraint (one for each initial s_t in {0,1} )   

RC0 = -exp(kp0) + ...
      (1/alf)*((1/bet)-1+del)*exp(z+alf*k) + ...
      (1-del)*exp(k) - ...
      ((1/alf)*((1/bet)-1+del)-del)*exp(c0);

RC1 = -exp(kp1) + ...
      (1/alf)*((1/bet)-1+del)*exp(z+alf*k) + ...
      (1-del)*exp(k) - ...
      ((1/alf)*((1/bet)-1+del)-del)*exp(c1); 

% residuals 
RC0x = jacobian(RC0,x); 
RC1x = jacobian(RC1,x); 

%% Save residual and jacobian 

% Flatten full system
R = [EE0;reshape(EE0x, [], 1);EE1;reshape(EE1x, [], 1);RC0;reshape(RC0x, [], 1);RC1;reshape(RC1x, [], 1)];

% Compute Jacobian w.r.t b
J = jacobian(R,b);

% Export function handles
R  = simplify(R, 'Steps', 50);
J  = simplify(J, 'Steps', 50);
Rh = matlabFunction(R, 'Vars', {x,b,p}, 'File', 'Res_eval', 'Optimize', true);
Jh = matlabFunction(J, 'Vars', {x,b,p}, 'File', 'Jac_eval', 'Optimize', true);

