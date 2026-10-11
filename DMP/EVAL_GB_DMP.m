function [q,qz,qzz] = EVAL_GB_DMP(globalDMP,z)

% (c) Eva F. Janssens and Sean McCrary
% These codes are part of the demo codes for the paper "Efficient Estimation 
% of Nonlinear DSGE Models" by Janssens & McCrary.

% Evaluate the analytic global solution and its first two derivatives.


zsize = size(z);
z = z(:);
a = globalDMP.a;
N = globalDMP.N;

x = polyval(flipud(a),z);

da = (1:N)'.*a(2:end);
dx = polyval(flipud(da),z);

d2a = (2:N)'.*(1:N-1)'.*a(3:end);
d2x = polyval(flipud(d2a),z);

q   = -log(x);
qz  = -dx./x;
qzz = (dx./x).^2-d2x./x;

q   = reshape(q,zsize);
qz  = reshape(qz,zsize);
qzz = reshape(qzz,zsize);
end
