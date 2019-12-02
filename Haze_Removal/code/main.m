%����ͼ��
input_image = imread('fog2.jpg'); %������ͼ��
figure, imshow(input_image),title('����ͼ��');

%dark channel����;
frame = 15; % ���ڴ�С
dark_channel=dc_process(frame, input_image);
figure, imshow(uint8(dark_channel)),title('����ͼ��İ�ͨ��');

%���A
Airlight = 170%max(dark_channel(:));   %��ͨ��ͼ��������ͼ�����ֵΪAֵ
%ֱ�ӵõ�͸����t
w = 0.95;%������Ϊ0.95
t = 1 - w * (dark_channel/Airlight);
figure,imshow(t,[]),title('ֱ�ӵõ��� t');

%ͨ��t�ָ���ͼ��
img_d = double(input_image);
J=recover(Airlight, t, img_d);
figure, imshow(uint8(J)),title('ͨ��t�ָ���ͼ���ͼ��');

%�˲���� t�밵ͨ��
r=60;
eps=10^-6;
t_d = guidedfilter(double(rgb2gray(input_image))/255, t, r, eps);
guide_dark_channel = (1-t_d)*Airlight/w;
figure, imshow(uint8(guide_dark_channel)),title('t_d�õ��İ�ͨ��');
figure,imshow(t_d,[]),title('�����˲���� t');

%ͨ��t_d�ָ���ͼ��
Jf = recover(Airlight, t_d, img_d);
figure,imshow(uint8(Jf)), title('ͨ��guide image filtering dark channel�ָ���ͼ��');

