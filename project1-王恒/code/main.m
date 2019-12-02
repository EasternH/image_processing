%输入图像
input_image = imread('fog2.jpg'); %待除雾图像
figure, imshow(input_image),title('输入图像');

%dark channel处理;
frame = 15; % 窗口大小
dark_channel=dc_process(frame, input_image);
figure, imshow(uint8(dark_channel)),title('输入图像的暗通道');

%获得A
Airlight = 170%max(dark_channel(:));   %暗通道图来从有雾图像最大值为A值
%直接得到透射率t
w = 0.95;%论文中为0.95
t = 1 - w * (dark_channel/Airlight);
figure,imshow(t,[]),title('直接得到的 t');

%通过t恢复的图像
img_d = double(input_image);
J=recover(Airlight, t, img_d);
figure, imshow(uint8(J)),title('通过t恢复的图像的图像');

%滤波后的 t与暗通道
r=60;
eps=10^-6;
t_d = guidedfilter(double(rgb2gray(input_image))/255, t, r, eps);
guide_dark_channel = (1-t_d)*Airlight/w;
figure, imshow(uint8(guide_dark_channel)),title('t_d得到的暗通道');
figure,imshow(t_d,[]),title('导向滤波后的 t');

%通过t_d恢复的图像
Jf = recover(Airlight, t_d, img_d);
figure,imshow(uint8(Jf)), title('通过guide image filtering dark channel恢复的图像');

