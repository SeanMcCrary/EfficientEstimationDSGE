function R = Res_eval(in1,in2,in3)
%Res_eval
%    R = Res_eval(IN1,IN2,IN3)

%    This function was generated by the Symbolic Math Toolbox version 24.2.
%    03-Jun-2025 21:59:03

b1 = in2(1,:);
b2 = in2(2,:);
b3 = in2(3,:);
b4 = in2(4,:);
b5 = in2(5,:);
b6 = in2(6,:);
b7 = in2(7,:);
b8 = in2(8,:);
p1 = in3(1,:);
p2 = in3(2,:);
p3 = in3(3,:);
p4 = in3(4,:);
p5 = in3(5,:);
p6 = in3(6,:);
p7 = in3(7,:);
p8 = in3(8,:);
p9 = in3(9,:);
p10 = in3(10,:);
p11 = in3(11,:);
p12 = in3(12,:);
p13 = in3(13,:);
p14 = in3(14,:);
p15 = in3(15,:);
p16 = in3(16,:);
p17 = in3(17,:);
p18 = in3(18,:);
p19 = in3(19,:);
p20 = in3(20,:);
p21 = in3(21,:);
p22 = in3(22,:);
p23 = in3(23,:);
p24 = in3(24,:);
p25 = in3(25,:);
p26 = in3(26,:);
p27 = in3(27,:);
p28 = in3(28,:);
p29 = in3(29,:);
p30 = in3(30,:);
x1 = in1(1,:);
x2 = in1(2,:);
x3 = in1(3,:);
t2 = exp(x1);
t3 = b3.*p5;
t4 = b2.*x1;
t5 = b3.*x2;
t6 = b4.*x3;
t7 = b6.*x1;
t8 = b7.*x2;
t9 = b8.*x3;
t10 = p4.*p9;
t11 = p4.*p10;
t12 = p4.*p11;
t13 = p4.*p12;
t14 = p4.*p13;
t15 = p4.*p14;
t16 = p4.*p15;
t17 = p4.*p16;
t18 = p4.*p17;
t19 = p4.*p18;
t20 = p4.*p19;
t21 = p3.*x3;
t22 = p4.*x3;
t23 = p7.*x1;
t27 = p3-1.0;
t28 = p7-1.0;
t29 = p8-1.0;
t30 = 1.0./p6;
t31 = 1.0./p7;
t26 = p1.*t5;
t32 = b4.*p5.*t10;
t33 = b4.*p5.*t11;
t34 = b4.*p5.*t12;
t35 = b4.*p5.*t13;
t36 = b4.*p5.*t14;
t37 = b4.*p5.*t15;
t38 = b4.*p5.*t16;
t39 = b4.*p5.*t17;
t40 = b4.*p5.*t18;
t41 = b4.*p5.*t19;
t42 = b4.*p5.*t20;
t43 = t23+x2;
t45 = t3-1.0;
t46 = p2.*t27;
t47 = p6.*t29;
t48 = t2.*t29;
t61 = t29+t30;
t66 = b1+t4+t5+t6;
t67 = b5+t7+t8+t9;
t44 = exp(t43);
t49 = -t32;
t50 = -t33;
t51 = -t34;
t52 = -t35;
t53 = -t36;
t54 = -t37;
t55 = -t38;
t56 = -t39;
t57 = -t40;
t58 = -t41;
t59 = -t42;
t60 = -t46;
t62 = t45.^2;
t63 = t47+1.0;
t64 = -t48;
t65 = p1.*t45.*x2;
t70 = exp(t66);
t71 = exp(t67);
t72 = b2.*t67;
t73 = p5.*t66;
t74 = t31.*t61;
t91 = t28.*t67;
t68 = -t65;
t69 = t22+t60;
t76 = -t73;
t78 = t10+t21+t60;
t79 = t11+t21+t60;
t80 = t12+t21+t60;
t81 = t13+t21+t60;
t82 = t14+t21+t60;
t83 = t15+t21+t60;
t84 = t16+t21+t60;
t85 = t17+t21+t60;
t86 = t18+t21+t60;
t87 = t19+t21+t60;
t88 = t20+t21+t60;
t89 = -t74;
t90 = b1+t72;
t107 = t44.*t74;
t75 = b4.*t69;
t92 = exp(t76);
t93 = exp(t78);
t94 = exp(t79);
t95 = exp(t80);
t96 = exp(t81);
t97 = exp(t82);
t98 = exp(t83);
t99 = exp(t84);
t100 = exp(t85);
t101 = exp(t86);
t102 = exp(t87);
t103 = exp(t88);
t105 = p8+t89;
t106 = p5.*t90;
t77 = p5.*t75;
t108 = -t106;
t109 = (t3.^2.*t93)./2.0;
t110 = (t3.^2.*t94)./2.0;
t111 = (t3.^2.*t95)./2.0;
t112 = (t3.^2.*t96)./2.0;
t113 = (t3.^2.*t97)./2.0;
t114 = (t3.^2.*t98)./2.0;
t115 = (t3.^2.*t99)./2.0;
t116 = (t3.^2.*t100)./2.0;
t117 = (t3.^2.*t101)./2.0;
t118 = (t3.^2.*t102)./2.0;
t119 = (t3.^2.*t103)./2.0;
t120 = (t62.*t93)./2.0;
t121 = (t62.*t94)./2.0;
t122 = (t62.*t95)./2.0;
t123 = (t62.*t96)./2.0;
t124 = (t62.*t97)./2.0;
t125 = (t62.*t98)./2.0;
t126 = (t62.*t99)./2.0;
t127 = (t62.*t100)./2.0;
t128 = (t62.*t101)./2.0;
t129 = (t62.*t102)./2.0;
t130 = (t62.*t103)./2.0;
t131 = t26+t75+t90;
t104 = -t77;
t132 = t49+t109;
t133 = t50+t110;
t134 = t51+t111;
t135 = t52+t112;
t136 = t53+t113;
t137 = t54+t114;
t138 = t55+t115;
t139 = t56+t116;
t140 = t57+t117;
t141 = t58+t118;
t142 = t59+t119;
t143 = p5.*t131;
t155 = t49+t120;
t156 = t50+t121;
t157 = t51+t122;
t158 = t52+t123;
t159 = t53+t124;
t160 = t54+t125;
t161 = t55+t126;
t162 = t56+t127;
t163 = t57+t128;
t164 = t58+t129;
t165 = t59+t130;
t144 = exp(t132);
t145 = exp(t133);
t146 = exp(t134);
t147 = exp(t135);
t148 = exp(t136);
t149 = exp(t137);
t150 = exp(t138);
t151 = exp(t139);
t152 = exp(t140);
t153 = exp(t141);
t154 = exp(t142);
t166 = exp(t155);
t167 = exp(t156);
t168 = exp(t157);
t169 = exp(t158);
t170 = exp(t159);
t171 = exp(t160);
t172 = exp(t161);
t173 = exp(t162);
t174 = exp(t163);
t175 = exp(t164);
t176 = exp(t165);
t177 = -t143;
t201 = t68+t91+t104+t108;
t178 = p20.*t144;
t179 = p21.*t145;
t180 = p22.*t146;
t181 = p23.*t147;
t182 = p24.*t148;
t183 = p25.*t149;
t184 = p26.*t150;
t185 = p27.*t151;
t186 = p28.*t152;
t187 = p29.*t153;
t188 = p30.*t154;
t189 = exp(t177);
t190 = p20.*t166;
t191 = p21.*t167;
t192 = p22.*t168;
t193 = p23.*t169;
t194 = p24.*t170;
t195 = p25.*t171;
t196 = p26.*t172;
t197 = p27.*t173;
t198 = p28.*t174;
t199 = p29.*t175;
t200 = p30.*t176;
t202 = exp(t201);
t203 = t178+t179+t180+t181+t182+t183+t184+t185+t186+t187+t188;
t204 = t190+t191+t192+t193+t194+t195+t196+t197+t198+t199+t200;
mt1 = [-t92-t47.*t189.*t203+t63.*t202.*t204;b2.*p5.*t92+t63.*t202.*t204.*(b6.*t28-b2.*b6.*p5)+b2.*b6.*p5.*t47.*t189.*t203;t3.*t92-t63.*t202.*t204.*(-b7.*t28+p1.*t45+b2.*b7.*p5)+p5.*t47.*t189.*t203.*(b2.*b7+b3.*p1)];
mt2 = [-t47.*t189.*(p3.*t109.*t178+p3.*t110.*t179+p3.*t111.*t180+p3.*t112.*t181+p3.*t113.*t182+p3.*t114.*t183+p3.*t115.*t184+p3.*t116.*t185+p3.*t117.*t186+p3.*t118.*t187+p3.*t119.*t188)+t63.*t202.*(p3.*t120.*t190+p3.*t121.*t191+p3.*t122.*t192+p3.*t123.*t193+p3.*t124.*t194+p3.*t125.*t195+p3.*t126.*t196+p3.*t127.*t197+p3.*t128.*t198+p3.*t129.*t199+p3.*t130.*t200)+b4.*p5.*t92-t63.*t202.*t204.*(-b8.*t28+b2.*b8.*p5+b4.*p4.*p5)+p5.*t47.*t189.*t203.*(b2.*b8+b4.*p4);t64-t71+t107+t70.*t105;t64-b6.*t71+t44.*t61+b2.*t70.*t105;t107-b7.*t71+b3.*t70.*t105;-b8.*t71+b4.*t70.*t105];
R = [mt1;mt2];
end
