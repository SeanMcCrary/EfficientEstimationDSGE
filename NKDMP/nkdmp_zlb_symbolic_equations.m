%-------------------------------------------------------------------------
% Description 
% This file constructs the residual functions of the NKDMP model taking
% three vectors as arguments 
% 1. p - vector of model parameters (both estimated and calibrated) 
% 2. x - vector of states 
% 3. b - vector of policy function coefficients 
%
% The file also constructs the analytic Jacobian as a function of these
% arguments. Both the residual function and Jacobian are defined using
% symbolic function then converted to Matlab functions that are called in
% the solution step of the estimation algorithm 
%
% NOTE!!! this file must be run before the solution/estimation routine 
% because it constructs the residual and jacobian functions 
%-------------------------------------------------------------------------

clear; close; clc; 
%-------------------------------------------------------------------------
% 0. Define Arguments 
syms x [6 1] real  % x = [n_t-1, z_t, d_t, m_t, s_t, a_t]  
syms b [28 1] real % b = [b10,..,b16,b20,...,b26,b30,...,b36,b40,...,b46]
% p      = [rz,sz,rd,sd,rm,sm,rs,ss,ra,sa,tau,tp,ty,kap,bet,alf,Qss,eta, c,gam,del,nss,nu,rss,lam]
% pindex = [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11,12,13, 14, 15, 16, 17, 18,19, 20, 21, 22,23, 24, 25]
syms p [25 1] real 

%-------------------------------------------------------------------------
% 0. Policies & Auxiliary Equations 

% policies 
tg = b(1)  + b(2)*x(1)  + b(3)*x(2)  + b(4)*x(3)  + b(5)*x(4)  + b(6)*x(5)  + b(7)*x(6);  % Market tightness 
in = b(8)  + b(9)*x(1)  + b(10)*x(2) + b(11)*x(3) + b(12)*x(4) + b(13)*x(5) + b(14)*x(6); % Inflation  
om = b(15) + b(16)*x(1) + b(17)*x(2) + b(18)*x(3) + b(19)*x(4) + b(20)*x(5) + b(21)*x(6); % Cost  
n =  b(22) + b(23)*x(1) + b(24)*x(2) + b(25)*x(3) + b(26)*x(4) + b(27)*x(5) + b(28)*x(6); % Employment   

% Taylor rule 
rs = p(12)*in + p(13)*(n+x(2)) + x(4); % shadow rate 
r  = (1/p(25))*log( exp(-p(25)*p(24) ) + exp(p(25)*rs) ); 
rx = jacobian(r,x); 

% Date t part of n_t+1-n_t 
Bn = b(23)*(n-x(1)) + ...
     b(24)*(p(1)-1)*x(2) + ...
     b(25)*(p(3)-1)*x(3) + ...
     b(26)*(p(5)-1)*x(4) + ...
     b(27)*(p(7)-1)*x(5) + ...
     b(28)*(p(9)-1)*x(6) ;

% Date t part of pi_t+1 
Bp  = b(8) + ...
      b(9)*n + ...
      b(10)*p(1)*x(2) + ...
      b(11)*p(3)*x(3) + ...
      b(12)*p(5)*x(4) + ...
      b(13)*p(7)*x(5) + ...
      b(14)*p(9)*x(6); 

% Date t part of tg_t+1
Btg = b(1) + ...
      b(2)*n + ...
      b(3)*p(1)*x(2) + ...
      b(4)*p(3)*x(3) + ...
      b(5)*p(5)*x(4) + ...
      b(6)*p(7)*x(5) + ...
      b(7)*p(9)*x(6); 

%-------------------------------------------------------------------------
% 1. Euler Equation

% x      = [n_t-1, z_t, d_t, m_t, s_t, a_t]  
%          [    1,   2,   3,   4,   5,   6] 
% b      = [b10,..,b16,b20,...,b26,b30,...,b36,b40,...,b46]
%          [  1,..., 7,  8,..., 14, 15,..., 21, 22,..., 28] 
% p      = [rz,sz,rd,sd,rm,sm,rs,ss,ra,sa,tau,tp,ty,kap,bet,alf,Qss,eta, c,gam,del,nss,nu,rss,lam]
% pindex = [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11,12,13, 14, 15, 16, 17, 18,19, 20, 21, 22,23, 24, 25]

EE = - 1 + (( exp(-p(25)*p(24)) + exp(p(25)*rs) )^(1/p(25)))...
   *exp(  (p(3)-1)*x(3) - p(11)*(p(1)-1)*x(2) - p(11)*Bn - Bp ... 
+ (1/2)*(  ((1-p(11)*b(25)  -b(11))^2)*(p(4)^2) ...
         + ((p(11)*(b(24)+1)+b(10))^2)*(p(2)^2) ...
         + ((p(11)*b(26)    +b(12))^2)*(p(6)^2)     ...
         + ((p(11)*b(27)    +b(13))^2)*(p(8)^2)     ...
         + ((p(11)*b(28)    +b(14))^2)*(p(10)^2) ) ); 

% residuals 
EEx = jacobian(EE,x); 

% %-------------------------------------------------------------------------
% % 2. Pricing Equation  

% x      = [n_t-1, z_t, d_t, m_t, s_t, a_t]  
%          [    1,   2,   3,   4,   5,   6] 
% b      = [b10,..,b16,b20,...,b26,b30,...,b36,b40,...,b46]
%          [  1,..., 7,  8,..., 14, 15,..., 21, 22,..., 28] 
% p      = [rz,sz,rd,sd,rm,sm,rs,ss,ra,sa,tau,tp,ty,kap,bet,alf,Qss,eta, c,gam,del,nss,nu,rss,lam]
% pindex = [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11,12,13, 14, 15, 16, 17, 18,19, 20, 21, 22,23, 24, 25]

PE = p(14)*(1-exp(om)) -(exp(in)-1)*exp(in) ...
     + p(15)*exp((p(3)-1)*x(3)+(1-p(11))*( (p(1)-1)*x(2) + Bn ) ) ...
     *( exp(2*Bp) ...
        *exp((1/2)*( ((1+(1-p(11))*b(25)  +2*b(11))^2)*(p(4)^2)  ... 
                   + (((1-p(11))*(b(24)+1)+2*b(10))^2)*(p(2)^2)  ...
                   + (((1-p(11))*b(26)    +2*b(12))^2)*(p(6)^2)  ...
                   + (((1-p(11))*b(27)    +2*b(13))^2)*(p(8)^2)  ...
                   + (((1-p(11))*b(28)    +2*b(14))^2)*(p(10)^2) )) ...
        -exp(Bp) ...
        *exp((1/2)*( ((1+(1-p(11))*b(25)  +b(11))^2)*(p(4)^2)  ... 
                 + ( ((1-p(11))*(b(24)+1) +b(10))^2)*(p(2)^2)  ...
                 + ( ((1-p(11))*b(26)     +b(12))^2)*(p(6)^2)  ...
                 + ( ((1-p(11))*b(27)     +b(13))^2)*(p(8)^2)  ...
                 + ( ((1-p(11))*b(28)     +b(14))^2)*(p(10)^2)  )) );

% residuals 
PEx = jacobian(PE,x); 


% %-------------------------------------------------------------------------
% % 3. Job-creation condition 
% x      = [n_t-1, z_t, d_t, m_t, s_t, a_t]  
%          [    1,   2,   3,   4,   5,   6] 
% b      = [b10,..,b16,b20,...,b26,b30,...,b36,b40,...,b46]
%          [  1,..., 7,  8,..., 14, 15,..., 21, 22,..., 28] 
% p      = [rz,sz,rd,sd,rm,sm,rs,ss,ra,sa,tau,tp,ty,kap,bet,alf,Qss,eta, c,gam,del,nss,nu,rss,lam]
% pindex = [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11,12,13, 14, 15, 16, 17, 18,19, 20, 21, 22,23, 24, 25]

JC = -exp(-x(6)+p(16)*tg) ...
    + (p(17)*(1-p(18))/p(19))*((1-p(20)*exp(om))*exp(x(2))-p(23)) ...
    +  p(15)*exp( (p(3)-1)*x(3) - p(11)*(p(1)-1)*x(2) - p(11)*Bn - p(9)*x(6) + p(16)*Btg )   ...
    *( exp( (1/2)*( ((p(16)*b(6)-p(11)*b(27))^2)*(p(8)^2) ))  ...
           - p(21)*exp(p(7)*x(5) + (1/2)*((p(16)*b(6)-p(11)*b(27)+1)^2)*(p(8)^2) ) ) ...
    *exp((1/2)*(  ((p(16)*b(4)-p(11)*b(25)+1)^2)*(p(4)^2) ...
               +  ((p(16)*b(3)-p(11)*b(24))^2)*(p(2)^2)   ...
               +  ((p(16)*b(5)-p(11)*b(26))^2)*(p(6)^2)   ...
               +  ((p(16)*b(7)-p(11)*b(28)-1)^2)*(p(10)^2) ));

% residuals 
JCx = jacobian(JC,x); 

%-------------------------------------------------------------------------
% 4. Law of motion 
% x      = [n_t-1, z_t, d_t, m_t, s_t, a_t]  
%          [    1,   2,   3,   4,   5,   6] 
% b      = [b10,..,b16,b20,...,b26,b30,...,b36,b40,...,b46]
%          [  1,..., 7,  8,..., 14, 15,..., 21, 22,..., 28] 
% p      = [rz,sz,rd,sd,rm,sm,rs,ss,ra,sa,tau,tp,ty,kap,bet,alf,Qss,eta, c,gam,del,nss,nu,rss,lam]
% pindex = [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11,12,13, 14, 15, 16, 17, 18,19, 20, 21, 22,23, 24, 25]

LM = -exp(n-x(1)) ...
     + 1-p(21)*exp(x(5)) ...
     + p(17)*exp(x(6)+(1-p(16))*tg)*((exp(-x(1))/p(22))-1+p(21)*exp(x(5)));

% residuals 
LMx = jacobian(LM,x); 

%% Save residuals and jacobian 

% Flatten full system
R = [EE;reshape(EEx, [], 1);PE;reshape(PEx, [], 1);JC;reshape(JCx, [], 1);LM;reshape(LMx, [], 1)];

% Compute Jacobian w.r.t b
J = jacobian(R,b);

% Export function handles
Rh = matlabFunction(R, 'Vars', {x,b,p}, 'File', 'Res_eval_zlb', 'Optimize', true);
Jh = matlabFunction(J, 'Vars', {x,b,p}, 'File', 'Jac_eval_zlb', 'Optimize', true);

% Export smoothed ZLB approximation 
NomRfun  = matlabFunction(r,  'Vars', {x,b,p}, 'File', 'NomR_zlb',    'Optimize', true);
NomRfunx = matlabFunction(rx, 'Vars', {x,b,p}, 'File', 'NomR_dx_zlb', 'Optimize', true);
