% This file tests the solution
clear; close; clc;
rng(1992)

make_plot = 0; 

% set parameters
hethh_parameters; 
 
%-------------------------------------------------------------------------%
%% Solve model 

tolout   = 1;
countout = 1;
maxit    = 50; 

tic 
while tolout>1e-6 && countout<=maxit % outer loop

   
    coeff  = aiyagari_cheby_inner(adim, par);
    Pu     = cheby_eval(agridFine,coeff(1:adim),amax);
    Pe     = cheby_eval(agridFine,coeff(adim+1:end),amax);
    apolu  = max(0,wu+(1+r-del)*agridFine - (exp(Pu).^(-1/tau)));
    apole  = max(0,we+(1+r-del)*agridFine - (exp(Pe).^(-1/tau)));


    % compute aggregates
    Pie = barycentric_weight_matrix(agridFine, apole);
    Piu = barycentric_weight_matrix(agridFine, apolu);

    % assemble Pi matrix
    Pi = [(1-f)*Piu    f*Piu;
        s*Pie     (1-s)*Pie];

    n = size(Pi,1);
    A = [Pi' - speye(n); ones(1,n)];
    d = [zeros(n,1); 1];
    pi = A\d;

    % CDFs
    Fu   = cumsum(pi(1:adimFine))/(1-L);
    Fe   = cumsum(pi(adimFine+1:end))/L;

    % capital holding by type
    Ku = estimate_E_Xk_from_CDF(agridFine,Fu,1);
    Ke = estimate_E_Xk_from_CDF(agridFine,Fe,1);
    K  = L*Ke + (1-L)*Ku;

    % check for convergence and update capital
    Ksupply  = K;
    Kdemand  = L*(alf/(r))^(1/(1 - alf));
    countout = countout+1;
    tolout   = norm(abs(Ksupply-Kdemand)./K);

    % Excess supply function
    excess = Ksupply - Kdemand;

    if excess > 0
        % Supply too high ⇒ lower r
        rmax = r;
    else
        % Supply too low ⇒ raise r
        rmin = r;
    end
    
    % update by bisection 
    r = 0.5*(rmin+rmax);

    w      = (1-alf)*(alf/(r))^(alf/(1 - alf)); 
    wu     = b*w; 
    we     = (1-tax)*w; 
    par.r  = r; 
    par.wu = wu; 
    par.we = we; 

    % reset inner
    countin = 0;
    tolin   = 1;

   
end
time=toc;  

fprintf('Outer Loop Iter %4.6f Error %4.6f Capital %4.2f Total Time %4.6f \n',countout,tolout,K,time)


%% plot final result

if make_plot ==1 

cpole = we+(1+r-del)*agridFine-apole; 
cpolu = wu+(1+r-del)*agridFine-apolu; 

figure(1);
plot(agridFine,cpole,'b','Linewidth',3); hold on;
plot(agridFine,cpolu,'r','Linewidth',3); hold off;

figure(2);
plot(agridFine,apole,'b','Linewidth',3); hold on;
plot(agridFine,apolu,'r','Linewidth',3); hold on;
plot(agridFine,agridFine,'-.k','Linewidth',1); hold off;

figure(3);
plot(agridFine,Fe,'b','Linewidth',3); hold on;
plot(agridFine,Fu,'r','Linewidth',3); hold off;

end 

%% Compute moments and approximations 
k  = 2;
mu = zeros(k,1);
me = zeros(k,1);

for ki=1:k
    mu(ki) = estimate_E_Xk_from_CDF(agridFine,Fu,ki);
    me(ki) = estimate_E_Xk_from_CDF(agridFine,Fe,ki);
end

% quadrature nodes and weights 
Nq     = 5;
tol    = 1;
crit   = 10^-8; 
maxit  = 11; 
au1    = [-2.80;0.50;0.02;-0.01;-0.0627;-0.0040];
ae1    = [-4.8; 1.5; -0.14; -0.01; 0.0642; -0.0043];
count  = 1; 
tolvec = zeros(maxit,1);

while tol>crit && count < maxit

    [ag, wg] = gauss_legendre_quadrature(Nq, 0, amax);

    au0     = au1(1:k);
    [gu,au] = maxent_pdf_kmoment_interval(mu, amax,au0,ag,wg);

    ae0 = ae1(1:k);
    [ge,ae] = maxent_pdf_kmoment_interval(me, amax,ae0,ag,wg);

    % check for convergence 
    tolvec(count) = norm(abs(au0-au(2:k+1))+abs(ae0-ae(2:k+1))); 
    tol           = tolvec(count); 
    count         = count+1; 
    Nq            = Nq+1; 
    ae1           = ae(2:k+1);
    au1           = au(2:k+1); 

end

if make_plot==1 

figure(5);
plot(agridFine,cumsum(gu(agridFine))*(agridFine(2)-agridFine(1)),'b','Linewidth',3); hold on;
plot(agridFine,Fu,'.r','Linewidth',3); hold on;
plot(agridFine,cumsum(ge(agridFine))*(agridFine(2)-agridFine(1)),'k','Linewidth',3); hold on;
plot(agridFine,Fe,'.r','Linewidth',3); hold off;

figure(6); 
plot(agridFine,gu(agridFine)./sum(gu(agridFine)),'r','Linewidth',3); hold on;
plot(agridFine,pi(1:adimFine)./sum(pi(1:adimFine)),'.r','Linewidth',3); hold off;

figure(7); 
plot(agridFine,ge(agridFine)./sum(ge(agridFine)),'b','Linewidth',3); hold on; 
plot(agridFine,pi(adimFine+1:end)./sum(pi(adimFine+1:end)),'.b','Linewidth',3); hold off;

figure(8);
plot(5:Nq,log10(tolvec(1:count)),'b','Linewidth',3); grid on;

end 

%------------------------------------------------------------------------------%
%------------------------------------------------------------------------------%
%------------------------------------------------------------------------------%
%------------------------------------------------------------------------------%
%------------------------------------------------------------------------------%
%% Auxiliary functions

function W = barycentric_weight_matrix(G, xvec)
% G: sorted 1D grid (n x 1 or 1 x n)
% xvec: query points (m x 1 or 1 x m)
% Returns W: m x n sparse matrix such that W(i,:) * G' ≈ xvec(i)
% Each row has only two nonzero entries with weights in [0,1]

G = G(:);              % Ensure column vector
xvec = xvec(:);        % Ensure column vector

n = length(G);
m = length(xvec);

% Step 1: Find left bracket index j such that G(j) <= x < G(j+1)
j = discretize(xvec, G);  % j in 1:(n-1), NaN for out-of-bounds

% Step 2: Clamp to boundary intervals
j(isnan(j) & xvec <= G(1))   = 1;
j(isnan(j) & xvec >= G(end)) = n - 1;

% Step 3: Compute raw interpolation weight
aL = G(j);         % G(j)
aR = G(j + 1);     % G(j+1)
t = (xvec - aL) ./ (aR - aL);  % interpolation fraction

% Step 4: Clamp t into [0,1] to prevent negative weights
t = min(max(t, 0), 1);  % ensures 0 ≤ t ≤ 1

% Step 5: Construct sparse matrix
row_idx = repmat((1:m)', 2, 1);      % [1;1;2;2;...;m;m]
col_idx = [j; j + 1];                % each pair of adjacent indices
vals    = [1 - t; t];                % barycentric weights, sum to 1

W = sparse(row_idx, col_idx, vals, m, n);
end

function EXk = estimate_E_Xk_from_CDF(x, F, k)
    % x: increasing grid (1D vector)
    % F: vector of CDF values F(x_i)
    % k: order of the moment
    if any(diff(x) <= 0)
        error('x must be strictly increasing.');
    end
    if k <= 0
        error('Moment order k must be positive.');
    end
    if F(end) < 0.999 
        warning('F(end) < 1; result may underestimate E[X^k].');
    end

    dx = diff(x);
    xL = x(1:end-1);
    xR = x(2:end);

    gL = xL.^(k - 1) .* (1 - F(1:end-1));
    gR = xR.^(k - 1) .* (1 - F(2:end));

    EXk = k * sum(0.5 * dx .* (gL + gR));
end

function [f_pdf, a_vec] = maxent_pdf_kmoment_interval(m_vec, a, a0, xg, wg)
% MAXENT_PDF_KMOMENT_INTERVAL
% Computes the max-entropy PDF on [0,a] matching raw moments m_vec
%
% Inputs:
%   m_vec : vector of raw moments [m1, m2, ..., mk]
%   a     : upper bound of support interval [0, a]
%   a0    : initial guess for coefficients [a1; ...; ak]
%   xg    : quadrature nodes in [0, a]
%   wg    : quadrature weights for nodes xg
%
% Outputs:
%   f_pdf : function handle for f(x) = exp(a0 + a1*x + ... + ak*x^k)
%   a_vec : coefficient vector [a0; a1; ...; ak], where a0 = -log(Z)

    % Solve for Lagrange multipliers (excluding normalization constant)
    opts = optimoptions('fsolve', 'Display', 'off', 'FunctionTolerance', 1e-10);
    a_body = fsolve(@(a) moment_residuals_k(a, m_vec, xg, wg), a0, opts);

    % Compute normalization constant
    expo = poly_exp(a_body, xg);
    Z    = sum(exp(expo) .* wg);
    a0   = -log(Z);

    % Output full parameter vector and function handle
    a_vec = [a0; a_body];
    f_pdf = @(x) (x >= 0 & x <= a) .* exp(a0 + poly_exp(a_body, x));

end


% === Helper: evaluates a1*x + a2*x^2 + ... + ak*x^k ===
function val = poly_exp(a_body, x)
    val = zeros(size(x));
    for j = 1:length(a_body)
        val = val + a_body(j) * x.^j;
    end
end

% === Helper: computes moment residuals ===
function F = moment_residuals_k(a_body, m_vec, x, w)
    fx = exp(poly_exp(a_body, x));
    Z = sum(fx .* w);
    F = zeros(length(m_vec), 1);
    for j = 1:length(m_vec)
        moment_j = sum(x.^j .* fx .* w) / Z;
        F(j) = moment_j - m_vec(j);
    end
end

% === Helper: Gauss–Legendre quadrature on [a,b] ===
function [x, w] = gauss_legendre_quadrature(N, a, b)
%GAUSS_LEGENDRE_QUADRATURE Compute Gauss–Legendre nodes and weights on [a, b]
%
%   [x, w] = gauss_legendre_quadrature(N, a, b)
%
% Inputs:
%   N - number of quadrature nodes
%   a - lower bound of integration interval
%   b - upper bound of integration interval
%
% Outputs:
%   x - Gauss–Legendre nodes in [a, b]
%   w - corresponding weights

    % Step 1: Build symmetric tridiagonal Jacobi matrix
    beta = 0.5 ./ sqrt(1 - (2*(1:N-1)).^(-2));
    T = diag(beta, 1) + diag(beta, -1);

    % Step 2: Eigenvalue decomposition (nodes are eigenvalues)
    [V, D] = eig(T);
    x_std = diag(D);              % nodes on [-1, 1]
    w_std = 2 * V(1,:).^2;        % weights on [-1, 1]

    % Step 3: Map to [a, b]
    x = 0.5 * (b - a) * x_std + 0.5 * (a + b);
    w = 0.5 * (b - a) * w_std';
end

