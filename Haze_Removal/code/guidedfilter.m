function q = guidedfilter(I, p, r, eps)
%   �����˲�O(1)
%
%   - I������ͼ�� (��һͨ�����߻�ֵͼ��)
%   - p: ���˲�ͼ�� (��һͨ�����߻�ֵͼ��)
%   - r: ���ڰ뾶
%   - eps: �ͷ�ֵ����ֹ�����Իع�ʱa����
%   ���䣺boxfilter�Ǵ�����ƽ���ĺ���

[hei, wid] = size(I);
N = boxfilter(ones(hei, wid), r); 

mean_I = boxfilter(I, r) ./ N;
mean_p = boxfilter(p, r) ./ N;
mean_Ip = boxfilter(I.*p, r) ./ N;
cov_Ip = mean_Ip - mean_I .* mean_p; % Э���� (I, p) .

mean_II = boxfilter(I.*I, r) ./ N;
var_I = mean_II - mean_I .* mean_I;

a = cov_Ip ./ (var_I + eps); % �����й�ʽ (5) 
b = mean_p - a .* mean_I; % �����й�ʽ (6) 

mean_a = boxfilter(a, r) ./ N;
mean_b = boxfilter(b, r) ./ N;

q = mean_a .* I + mean_b; % �����й�ʽ (8) 
end