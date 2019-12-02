function q = guidedfilter(I, p, r, eps)
%   导向滤波O(1)
%
%   - I：引导图像 (单一通道或者灰值图像)
%   - p: 带滤波图像 (单一通道或者灰值图像)
%   - r: 窗口半径
%   - eps: 惩罚值，防止在线性回归时a过大
%   补充：boxfilter是窗口求平均的函数

[hei, wid] = size(I);
N = boxfilter(ones(hei, wid), r); 

mean_I = boxfilter(I, r) ./ N;
mean_p = boxfilter(p, r) ./ N;
mean_Ip = boxfilter(I.*p, r) ./ N;
cov_Ip = mean_Ip - mean_I .* mean_p; % 协方差 (I, p) .

mean_II = boxfilter(I.*I, r) ./ N;
var_I = mean_II - mean_I .* mean_I;

a = cov_Ip ./ (var_I + eps); % 论文中公式 (5) 
b = mean_p - a .* mean_I; % 论文中公式 (6) 

mean_a = boxfilter(a, r) ./ N;
mean_b = boxfilter(b, r) ./ N;

q = mean_a .* I + mean_b; % 论文中公式 (8) 
end