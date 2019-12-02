function imDst = boxfilter(imSrc, r)

%   BOXFILTER   ������ʹ���Ӷ�ΪO(MN)����ͣ��󷽲�����㽵�͵�O(1)�������O(1)�ĸ��Ӷ�
%   imSrc :����ͼ��
%   r: �����뾶
%   ������Ĵ�������ʾ�������� imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
%   �����ٶȻ����


[hei, wid] = size(imSrc);
imDst = zeros(size(imSrc));

%��Y���ۼ�
imCum = cumsum(imSrc, 1);
%���ȿ����ײ���r������
imDst(1:r+1, :) = imCum(1+r:2*r+1, :);
%�м�����
imDst(r+2:hei-r, :) = imCum(2*r+2:hei, :) - imCum(1:hei-2*r-1, :);
%β��r������
imDst(hei-r+1:hei, :) = repmat(imCum(hei, :), [r, 1]) - imCum(hei-2*r:hei-r-1, :);

%��X���ۼ�
imCum = cumsum(imDst, 2);
%���ȿ����ײ���r������
imDst(:, 1:r+1) = imCum(:, 1+r:2*r+1);
%�����м�����
imDst(:, r+2:wid-r) = imCum(:, 2*r+2:wid) - imCum(:, 1:wid-2*r-1);
%����β��r������
imDst(:, wid-r+1:wid) = repmat(imCum(:, wid), [1, r]) - imCum(:, wid-2*r:wid-r-1);
end

