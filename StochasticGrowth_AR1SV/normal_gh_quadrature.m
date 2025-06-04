function [z, omega] = normal_gh_quadrature(N)
%NORMAL_GH_QUADRATURE Gauss-Hermite nodes and weights for standard normal integrals.
%
%   [z, omega] = normal_gh_quadrature(N) returns N nodes and weights for
%   approximating integrals of the form ∫ f(x) φ(x) dx, where φ(x) is the
%   standard normal density.
%
%   Inputs:
%     N      - Number of quadrature points
%
%   Outputs:
%     z      - Nodes (N×1), scaled for N(0,1)
%     omega  - Weights (N×1), sum to 1
%
%   Example:
%     [z, w] = normal_gh_quadrature(5);
%     approx = sum(w .* z.^2);  % Approximates E[x^2] = 1

% Step 1: Build Hermite companion matrix (Golub-Welsch algorithm)
i = (1:N-1)';
a = zeros(N, 1);        % Zero diagonal
b = sqrt(i/2);          % Subdiagonal
J = diag(a) + diag(b,1) + diag(b,-1);

% Step 2: Compute eigenvalues and eigenvectors
[V, D] = eig(J);
x      = diag(D);              % Classical Hermite nodes (for e^{-x^2})
w      = V(1,:).^2 * sqrt(pi); % Classical Hermite weights (for e^{-x^2})

% Step 3: Convert to standard normal nodes and weights
z     = x*sqrt(2);      % Nodes for standard normal (e^{-x^2/2})
omega = w/sqrt(pi);     % Weights for standard normal (e^{-x^2/2})

% Sort nodes and weights
[z, sort_idx] = sort(z);
omega         = omega(sort_idx);

end
