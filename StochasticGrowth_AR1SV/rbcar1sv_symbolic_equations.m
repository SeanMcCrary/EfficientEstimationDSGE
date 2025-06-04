%-------------------------------------------------------------------------
% Description 
% This file constructs the residual functions of the stochastic growth
% model with markov switching taking three vectors as arguments 
% 1. p - vector of model parameters (including integration nodes and weights)  
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
syms x [3 1] real  % x = [k_t, z_t, s_t] 

% policy coefficients 
syms b [8 1] real % b = [a0,a1,a2,a3,b0,b1,b2,b3]

% parameters (note this assumes N=5 for integration nodes and weights) 
syms p [30 1] real  % p      = [rz,mu,rx,sx,tau,bet,alf,del,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11]
                    % pindex = [ 1, 2, 3, 4,  5,  6,  7,  8, 9,10,11,12,13,14,15,16,17,18, 19, 20,21,22,23,24,25,26,27,28, 29, 30] 


%-------------------------------------------------------------------------
% 0. Policies & Auxiliary Equations 

% parameters 
rz  = p(1);  mu  = p(2);  rx  = p(3);  sx  = p(4);                   % AR(1)-SV
tau = p(5);  bet = p(6);  alf = p(7);  del = p(8);                   % preferences & technology
n1  = p(9);  n2  = p(10); n3  = p(11); n4  = p(12); n5  = p(13);     % integration nodes and weights 
n6  = p(14); n7  = p(15); n8  = p(16); n9  = p(17); n10 = p(18); n11 = p(19);
w1  = p(20); w2  = p(21); w3  = p(22); w4  = p(23); w5  = p(24);
w6  = p(25); w7  = p(26); w8  = p(27); w9  = p(28); w10 = p(29); w11 = p(30);

% states k_t, z_t, x_t
k = x(1); z = x(2); s = x(3); 

% policy coefficient (mapping to notes) 
a0 = b(1);  a1 = b(2);  a2 = b(3); a3 = b(4);  
b0 = b(5);  b1 = b(6);  b2 = b(7); b3 = b(8);

% policies: consumption c_t and capital k_t+1 
c  = a0 + a1*k + a2*z + a3*s; 
kp = b0 + b1*k + b2*z + b3*s;

% constant terms factored out of expectations 
g0 = exp(-tau*(a0+a1*kp)+(alf-1)*kp+(1-tau*a2)*rz*z-tau*a3*((1-rx)*mu+sx*s));

h0 = exp(-tau*(a0+a1*kp+a2*rz*z+a3*((1-rx)*mu+sx*s)));

% expectation terms for Euler equations (Gauss-Hermite quadrature) 
g1 = w1*exp( -tau*a3*sx*n1+((1-tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n1)) + ... 
     w2*exp( -tau*a3*sx*n2+((1-tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n2)) + ... 
     w3*exp( -tau*a3*sx*n3+((1-tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n3)) + ... 
     w4*exp( -tau*a3*sx*n4+((1-tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n4)) + ... 
     w5*exp( -tau*a3*sx*n5+((1-tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n5)) + ...
     w6*exp( -tau*a3*sx*n6+((1-tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n6)) + ...
     w7*exp( -tau*a3*sx*n7+((1-tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n7)) + ...
     w8*exp( -tau*a3*sx*n8+((1-tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n8)) + ...
     w9*exp( -tau*a3*sx*n9+((1-tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n9)) + ...
     w10*exp(-tau*a3*sx*n10+((1-tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n10)) + ...    
     w11*exp(-tau*a3*sx*n11+((1-tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n11)); 

h1 = w1*exp( -tau*a3*sx*n1+((tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n1)) + ... 
     w2*exp( -tau*a3*sx*n2+((tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n2)) + ... 
     w3*exp( -tau*a3*sx*n3+((tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n3)) + ... 
     w4*exp( -tau*a3*sx*n4+((tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n4)) + ... 
     w5*exp( -tau*a3*sx*n5+((tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n5)) + ... 
     w6*exp( -tau*a3*sx*n6+((tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n6)) + ... 
     w7*exp( -tau*a3*sx*n7+((tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n7)) + ... 
     w8*exp( -tau*a3*sx*n8+((tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n8)) + ...      
     w9*exp( -tau*a3*sx*n9+((tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n9)) + ... 
     w10*exp(-tau*a3*sx*n10+((tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n10)) + ...     
     w11*exp(-tau*a3*sx*n11+((tau*a2)^2/2)*exp((1-rx)*mu+rx*s+sx*n11));

%-------------------------------------------------------------------------
% 1. Euler Equations  

EE  = -exp(-tau*c) + ...
      (1-bet*(1-del))*(g0*g1) + ...
      bet*(1-del)*(h0*h1);

% residuals 
EEx = jacobian(EE,x); 

%-------------------------------------------------------------------------
% 2. Resource constraint    

RC  = -exp(kp) + ...
      (1/alf)*((1/bet)-1+del)*exp(z+alf*k) + ...
      (1-del)*exp(k) - ...
      ((1/alf)*((1/bet)-1+del)-del)*exp(c);

% residuals 
RCx = jacobian(RC,x); 

%% Save residual and jacobian 

% Flatten full system
R = [EE;reshape(EEx, [], 1);RC;reshape(RCx, [], 1)];

% Compute Jacobian w.r.t b
J = jacobian(R,b);

% Export function handles
Rh = matlabFunction(R, 'Vars', {x,b,p}, 'File', 'Res_eval', 'Optimize', true);
Jh = matlabFunction(J, 'Vars', {x,b,p}, 'File', 'Jac_eval', 'Optimize', true);

