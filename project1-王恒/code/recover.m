function re= recover(A, t, I)
%   恢复函数
%
%   - A: 是Airlight
%   - t: 是透射率
%   - I: 是观测到的有雾图像
%   补:设定了一个t阈值0.1防止t过小

J(:,:,1) = (I(:,:,1)- A)./max(t,0.1) + A;
J(:,:,2) = (I(:,:,2)- A)./max(t,0.1) + A;
J(:,:,3) = (I(:,:,3)- A)./max(t,0.1) + A;

re=J;

end