function imDst = boxfilter(imSrc, r)

%   BOXFILTER   它可以使复杂度为O(MN)的求和，求方差等运算降低到O(1)或近似于O(1)的复杂度
%   imSrc :输入图像
%   r: 滑窗半径
%   用下面的代码来表示上述过程 imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
%   但是速度会更快


[hei, wid] = size(imSrc);
imDst = zeros(size(imSrc));

%按Y轴累加
imCum = cumsum(imSrc, 1);
%首先考虑首部的r个像素
imDst(1:r+1, :) = imCum(1+r:2*r+1, :);
%中间像素
imDst(r+2:hei-r, :) = imCum(2*r+2:hei, :) - imCum(1:hei-2*r-1, :);
%尾部r个像素
imDst(hei-r+1:hei, :) = repmat(imCum(hei, :), [r, 1]) - imCum(hei-2*r:hei-r-1, :);

%按X轴累加
imCum = cumsum(imDst, 2);
%首先考虑首部的r个像素
imDst(:, 1:r+1) = imCum(:, 1+r:2*r+1);
%考虑中间像素
imDst(:, r+2:wid-r) = imCum(:, 2*r+2:wid) - imCum(:, 1:wid-2*r-1);
%考虑尾部r个像素
imDst(:, wid-r+1:wid) = repmat(imCum(:, wid), [1, r]) - imCum(:, wid-2*r:wid-r-1);
end

