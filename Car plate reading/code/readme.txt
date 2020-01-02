crop40:
该文件夹是我们对原图经过裁剪后得到的单个车牌图像。其中文件命名规则为：
cropimg_x.jpg，x表示裁剪的车牌序号。
例如
cropimg_0.jpg cropimg_1.jpg ... cropimg_40.jpg


character40:
该文件夹是我们对crop40文件夹中的每一个单个车牌进行字符分割的结果。其中文件命名规则为：
cropimg_x_y.jpg，其中x，y为变量，分别表示第x个车牌和在第x个车牌中分割的第y个字符。
例如
cropimg_0_0.jpg  表示对cropimg_0.jpg进行车牌分割得到的第0个字符
cropimg_1_2.jpg 表示对cropimg_1.jpg进行车牌分割得到的第2个字符