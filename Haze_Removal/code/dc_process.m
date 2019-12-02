function dark_cha = dc_process(frame, input_image)

% 获得图像的暗通道
% frame: 窗口半径
% input_image: 输入的图像


%预分配内存
Three_D_min=input_image;
%每个通道进行滑窗
for k = 1 : 3
    minimum_intense = ordfilt2((input_image(:, :, k)), 1, ones(frame), 'symmetric');%二维统计顺序滤波 15*15的框里找最小值
    Three_D_min(:,:,k)  = minimum_intense;
end

%全通道的暗通道
[h,w,z]=size(input_image);
dark_cha = zeros(h,w);
for y=1:h
    for x=1:w
        dark_cha(y,x) = min(Three_D_min(y,x,:));
    end
end

end
