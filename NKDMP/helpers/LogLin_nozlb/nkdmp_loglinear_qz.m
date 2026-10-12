function out = nkdmp_loglinear_qz(par)
%--------------------------------------------------------------------------
% Solve the log-linear NK-DMP model by generalized Schur (QZ) decomposition.
%
% State:
%   x_t = [n_{t-1}, z_t, d_t, m_t, s_t, a_t]'
%
% Controls:
%   y_t = [n_t, pi_t, omega_t, q_t, theta_t]'
%
% Policy rule:
%   y_t = P x_t
%
% INPUT
%   par : struct with fields (aliases allowed)
%         rz, rd, rm, rs, ra
%         tau, phipi, phiy, kap, bet, alf
%         Q, chi, gam, del, N
%
%         Allowed aliases:
%           kap or kappa
%           bet or beta
%           alf or alpha
%           Q   or Qss
%           gam or gamma
%           del or delta
%           N   or Npop or nss
%
% OUTPUT
%   out.P         : 5x6 policy matrix
%   out.M         : 6x6 state transition matrix M = Rbar + H*P
%   out.residual  : Frobenius norm of A0*P - A1*P*(Rbar+H*P) - B
%   out.eig_qz    : reordered generalized eigenvalues
%   out.nstable   : number of stable roots selected
%   out.unique    : true if exactly nx stable roots and Zx invertible
%   out.b_tp      : 28x1 coefficient vector usable as steady-state TP guess
%                   order = [theta; pi; omega; n], each with zero intercept
%--------------------------------------------------------------------------
    
    %-----------------------------
    % 1. Read parameters
    rz    = fetch(par, {'rz','rho_z'});
    rd    = fetch(par, {'rd','rho_d'});
    rm    = fetch(par, {'rm','rho_m'});
    rs    = fetch(par, {'rs','rho_s'});
    ra    = fetch(par, {'ra','rho_a'});
    
    tau   = fetch(par, {'tau'});
    phipi = fetch(par, {'phipi','phi_pi'});
    phiy  = fetch(par, {'phiy','phi_y'});
    kap   = fetch(par, {'kap','kappa'});
    bet   = fetch(par, {'bet','beta'});
    alf   = fetch(par, {'alf','alpha'});
    
    Q     = fetch(par, {'Q','Qss'});
    chi   = fetch(par, {'chi'});
    gam   = fetch(par, {'gam','gamma'});
    del   = fetch(par, {'del','delta'});
    N     = fetch(par, {'N','Npop','nss'});
    
    %-----------------------------
    % 2. Build matrices from appendix
    
    % Dimensions
    nx = 6;
    ny = 5;
    
    % R and Rbar
    R = diag([rz, rd, rm, rs, ra]);
    Rbar = [0, zeros(1,5);
            zeros(5,1), R];
    
    % H = e1 * jn
    e1 = [1;0;0;0;0;0];
    jn = [1,0,0,0,0];
    H  = e1 * jn;   % 6x5
    
    % A0, A1, B
    A0 = [tau + phiy,          phipi,      0,              0,          0;
          0,                   1,          kap,            0,          0;
          bet*(1-del)*tau,     0,         -chi*gam,        1,          0;
          1,                   0,          0,             -del,       -del;
          0,                   0,          0,              1,          alf];
    
    A1 = [tau,                 1,          0,              0,              0;
          0,                   bet,        0,              0,              0;
          bet*(1-del)*tau,     0,          0,              bet*(1-del),    0;
          0,                   0,          0,              0,              0;
          0,                   0,          0,              0,              0];
    
    B  = [0, tau*(rz-1)-phiy,                           1-rd,                  -1,             0,              0;
          0, 0,                                         0,                     0,              0,              0;
          0, -chi*(1-gam)+bet*(1-del)*tau*(rz-1),      bet*(1-del)*(1-rd),    0,      bet*del*rs,          0;
          1-Q/N, 0,                                     0,                     0,     -del*(1-Q),           0;
          0, 0,                                         0,                     0,              0,              1];
    
    % Stacked LRE system:
    % Gamma0 * E_t[xi_{t+1}] = Gamma1 * xi_t
    Gamma0 = [eye(nx),         zeros(nx,ny);
              zeros(ny,nx),    A1];
    
    Gamma1 = [Rbar,            H;
              -B,              A0];
    
    %-----------------------------
    % 3. QZ decomposition and stable reordering
    
    [AA,BB,Qq,Z] = qz(Gamma1, Gamma0, 'real');
    [AA,BB,~,Z]  = ordqz(AA,BB,Qq,Z,'udi');
    
    eig_qz = ordeig(AA,BB);
    
    qz_tol    = 1e-9;
    is_stable = isfinite(eig_qz) & (abs(eig_qz) < 1 - qz_tol);
    nstable   = sum(is_stable);
    
    % Need exactly nx stable roots
    if nstable ~= nx
        error(['Blanchard-Kahn condition failed: expected %d stable roots, ', ...
               'found %d.'], nx, nstable);
    end
    
    %-----------------------------
    % 4. Extract stable invariant subspace
    
    Zs = Z(:,1:nx);
    Zx = Zs(1:nx,:);
    Zy = Zs(nx+1:end,:);
    
    if rcond(Zx) < 1e-12
        error('Stable-state block Zx is singular or ill-conditioned.');
    end
    
    P = real(Zy / Zx);
    M = Rbar + H*P;
    
    % Residual check
    residual = norm(A0*P - A1*P*(Rbar + H*P) - B, 'fro');
    
    %-----------------------------
    % 5. TP-style steady-state guess (zero intercepts)
    %
    % TP order:
    % [theta0, theta slopes(6),
    %  pi0,    pi slopes(6),
    %  omega0, omega slopes(6),
    %  n0,     n slopes(6)]'
    
    b_tp = [0; P(5,:)'; ...
            0; P(2,:)'; ...
            0; P(3,:)'; ...
            0; P(1,:)'];
    
    %-----------------------------
    % 6. Output
    out = struct();
    out.P        = P;
    out.M        = M;
    out.residual = residual;
    out.eig_qz   = eig_qz;
    out.nstable  = nstable;
    out.unique   = (nstable == nx) && (rcond(Zx) > 1e-12);
    out.b_tp     = b_tp;
    
    out.rows = struct();
    out.rows.n     = P(1,:);
    out.rows.pi    = P(2,:);
    out.rows.omega = P(3,:);
    out.rows.q     = P(4,:);
    out.rows.theta = P(5,:);
end

%==========================================================================
% Helper: fetch parameter with aliases
%==========================================================================
function val = fetch(s, names)
    for i = 1:numel(names)
        if isfield(s, names{i})
            val = s.(names{i});
            return
        end
    end
    error('Missing parameter. Tried aliases: %s', strjoin(names, ', '));
end