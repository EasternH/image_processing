function dark_cha = dc_process(frame, input_image)

% ���ͼ��İ�ͨ��
% frame: ���ڰ뾶
% input_image: �����ͼ��


%Ԥ�����ڴ�
Three_D_min=input_image;
%ÿ��ͨ�����л���
for k = 1 : 3
    minimum_intense = ordfilt2((input_image(:, :, k)), 1, ones(frame), 'symmetric');%��άͳ��˳���˲� 15*15�Ŀ�������Сֵ
    Three_D_min(:,:,k)  = minimum_intense;
end

%ȫͨ���İ�ͨ��
[h,w,z]=size(input_image);
dark_cha = zeros(h,w);
for y=1:h
    for x=1:w
        dark_cha(y,x) = min(Three_D_min(y,x,:));
    end
end

end
