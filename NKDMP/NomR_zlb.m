function r = NomR_zlb(in1,in2,in3)
%NomR_zlb
%    R = NomR_zlb(IN1,IN2,IN3)

%    This function was generated by the Symbolic Math Toolbox version 24.2.
%    29-May-2025 15:48:11

b8 = in2(8,:);
b9 = in2(9,:);
b10 = in2(10,:);
b11 = in2(11,:);
b12 = in2(12,:);
b13 = in2(13,:);
b14 = in2(14,:);
b22 = in2(22,:);
b23 = in2(23,:);
b24 = in2(24,:);
b25 = in2(25,:);
b26 = in2(26,:);
b27 = in2(27,:);
b28 = in2(28,:);
p12 = in3(12,:);
p13 = in3(13,:);
p24 = in3(24,:);
p25 = in3(25,:);
x1 = in1(1,:);
x2 = in1(2,:);
x3 = in1(3,:);
x4 = in1(4,:);
x5 = in1(5,:);
x6 = in1(6,:);
r = log(exp(p25.*(x4+p13.*(b22+x2+b23.*x1+b24.*x2+b25.*x3+b26.*x4+b27.*x5+b28.*x6)+p12.*(b8+b9.*x1+b10.*x2+b11.*x3+b12.*x4+b13.*x5+b14.*x6)))+exp(-p24.*p25))./p25;
end
