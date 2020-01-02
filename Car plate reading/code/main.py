"""
主要的步骤为：
1）提取单通道图片，选项为 （灰度图片/HSV中的value分支）
2）提升对比度，选项为 （形态学中的顶帽/灰度拉伸）
3）边缘连接（膨胀）
4）二值化
5）利用findcontours函数找到边缘
6）裁剪图片，车牌图片存储
7) 对车牌预处理
8）方向矫正
9）车牌精确区域搜索
10） 字符分割
11） 字符识别
"""

import cv2
import copy
import numpy as np
import math
import os

def SingleChannel(img) :
    """
    用于车牌检测
    得到单通道图片，主要测试两种方式，灰度通道以及hsv中的v通道
    :param img: 输入图片
    :return:
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    cv2.imshow("SingleChannel", value)
    return value

def Contrast(img) :
    """
    用于车牌检测
    利用tophat，提高图片对比度,
    :param img: 输入图片
    :return:
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # applying topHat/blackHat operations
    topHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    cv2.imshow("tophat", topHat)
    blackHat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    cv2.imshow("blackhat", blackHat)
    add = cv2.add(img, topHat)
    subtract = cv2.subtract(add, blackHat)
    cv2.imshow('Constrast', subtract)
    return subtract

def threshold(img) :
    """
    用于车牌检测
    采用cv2.adaptiveThreshold方法，对图片二值化
    :param img: 输入图像
    :return:
    """
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    cv2.imshow("thresh", thresh)
    return thresh

def Contrast_2(img) :
    """
    用于字符识别，对单个车牌图像预处理
    :param img:  输入图片
    :return:
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)
    add = cv2.add(value, topHat)
    subtract = cv2.subtract(add, blackHat)
    #cv2.imshow('subtract', subtract)
    return subtract

def threshold_2(img) :
    """
    用于字符识别，对单个车牌图像预处理
    :param img: 输入图片
    :return:
    """
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    return thresh
# 存储裁剪车牌图像序号，cropimg_0.jpg,cropimg_1.jpg...
global crop_num
crop_num = 0

def drawCoutrous(img_temp) :
    """
    对输入图像查找内边缘，设置阈值，去除一些面积较小的内边缘
    :param img_temp: 输入图像，经过预处理
    :return:
    """
    threshline = 2000
    imgCopy = copy.deepcopy(img_temp)
    contours, hierarchy = cv2.findContours(imgCopy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours), contours[0].shape)
    # print(hierarchy.shape)
    maxarea = 0
    conid = 0
    img_zero = np.zeros(img.shape)
    # print("img_zero.shape is : ",img_zero.shape)
    num_contours = 0
    contoursList = []
    for i in range(len(contours)) :
        if hierarchy[0][i][3] >= 0 :
            temparea = math.fabs(cv2.contourArea(contours[i]))
            # print(math.fabs(cv2.contourArea(contours[i])))
            if temparea > maxarea :
                conid = i
                maxarea = temparea
            if temparea > threshline :
                num_contours += 1
                if num_contours % 7 == 0 :
                    cv2.drawContours(img_zero, contours, i, (0,0,255),1)
                if num_contours % 7 == 1 :
                    cv2.drawContours(img_zero, contours, i, (255,0,0),1)
                if num_contours % 7 == 2 :
                    cv2.drawContours(img_zero, contours, i, (0,255,0),1)
                if num_contours % 7 == 3 :
                    cv2.drawContours(img_zero, contours, i, (0,255,255),1)
                if num_contours % 7 == 4 :
                    cv2.drawContours(img_zero, contours, i, (255,0,255),1)
                if num_contours % 7 == 5 :
                    cv2.drawContours(img_zero, contours, i, (255,255,0),1)
                if num_contours % 7 == 6:
                    cv2.drawContours(img_zero, contours, i, (255, 255, 255), 1)
                # print(contours[i].shape)
                contoursList.append(contours[i])
    # print("maxarea: ",maxarea)
    # print("number of contours is ", num_contours)
    # cv2.drawContours(img_zero, contours, conid, (0, 0, 255), 1)
    cv2.imshow("with contours",img_zero)
    return contoursList

def DrawRectangle(img, img_temp, ConList) :
    """
    得到车牌边缘的的x，y坐标最小最大值，再原图上绘制bounding box，得到裁剪后的车牌图像
    :param img:      原图
    :param img_temp:    二值图像
    :param ConList:     图像的边缘轮廓
    :return:   null
    """
    length = len(ConList)
    rectanglePoint = np.zeros((length, 4, 1, 2), dtype = np.int32)
    img_zeros = np.zeros(img_temp.shape)
    img_copy = copy.deepcopy(img)
    img_copy_1 = copy.deepcopy(img)
    # print("img_zeros, length; ", img_zeros.shape, length)
    for i in range(length) :
        contours = ConList[i]
        minx, maxx, miny, maxy = 1e6, 0, 1e6, 0
        for index_num in range(contours.shape[0]) :
            if contours[index_num][0][0] < minx :
                minx = contours[index_num][0][0]
            if contours[index_num][0][0] > maxx :
                maxx = contours[index_num][0][0]
            if contours[index_num][0][1] < miny :
                miny = contours[index_num][0][1]
            if contours[index_num][0][1] > maxy :
                maxy = contours[index_num][0][1]
        # print(minx, maxx, miny, maxy)
        rectanglePoint[i][0][0][0], rectanglePoint[i][0][0][1] = minx, miny
        rectanglePoint[i][1][0][0], rectanglePoint[i][1][0][1] = minx, maxy
        rectanglePoint[i][2][0][0], rectanglePoint[i][2][0][1] = maxx, maxy
        rectanglePoint[i][3][0][0], rectanglePoint[i][3][0][1] = maxx, miny
        # rectanglePoint.dtype = np.int32
        # print(rectanglePoint[i].shape)
        crop_save(minx, maxx, miny, maxy, img_copy_1)
        # print("dx: ",maxx-minx,"dy: ",maxy-miny, "area: ", (maxx-minx)*(maxy-miny))
        cv2.polylines(img_copy, [rectanglePoint[i]], True, (0,0,255),2)
    cv2.imshow("img_zeros_haha", img_copy)

def crop_save(minx, maxx, miny, maxy, img_original) :
    """
    裁剪原图，根据minx，maxx，miny，maxy
    :param minx: x坐标最小值
    :param maxx: x坐标最大值
    :param miny: y坐标最小值
    :param maxy: y坐标最大值
    :param img_original: 由于需要将绘制结果再原图中显示，输入原图
    :return:
    """
    global crop_num
    epsx = 60
    epsy = 30
    dx = maxx - minx
    dy = maxy - miny
    if dx == dy :
        return
    if dx >= 600 - epsx :
        dx1, dx2, dx3, dx4 = minx, minx + 1 * int(dx / 3), minx + 2 * int(dx / 3), maxx
        save_pth = './crop40/cropimg_' + str(crop_num) + '.jpg'
        # cv2.imwrite(save_pth, img_original[dx1:dx2, miny:maxy,:])
        cv2.imwrite(save_pth, img_original[miny:maxy, dx1:dx2, :])
        crop_num += 1
        save_pth = './crop40/cropimg_' + str(crop_num) + '.jpg'
        cv2.imwrite(save_pth, img_original[miny:maxy, dx2:dx3, :])
        crop_num += 1
        save_pth = './crop40/cropimg_' + str(crop_num) + '.jpg'
        cv2.imwrite(save_pth, img_original[miny:maxy, dx3:dx4, :])
        crop_num += 1
    elif dx >= 400 - epsx :
        dx1, dx2, dx3 = minx, minx + 1 * int(dx / 2), maxx
        save_pth = './crop40/cropimg_' + str(crop_num) + '.jpg'
        cv2.imwrite(save_pth, img_original[miny:maxy, dx1:dx2, :])
        crop_num += 1
        save_pth = './crop40/cropimg_' + str(crop_num) + '.jpg'
        cv2.imwrite(save_pth, img_original[miny:maxy, dx2:dx3, :])
        crop_num += 1
    elif dy >= 240 - epsy :
        dy1, dy2, dy3, dy4 = miny, miny + 1 * int(dy / 3), miny + 2 * int(dy / 3), maxy
        save_pth = './crop40/cropimg_' + str(crop_num) + '.jpg'
        cv2.imwrite(save_pth, img_original[dy1: dy2, minx:maxx, :])
        crop_num += 1
        save_pth = './crop40/cropimg_' + str(crop_num) + '.jpg'
        cv2.imwrite(save_pth, img_original[dy2: dy3, minx:maxx, :])
        crop_num += 1
        save_pth = './crop40/cropimg_' + str(crop_num) + '.jpg'
        cv2.imwrite(save_pth, img_original[dy3: dy4, minx:maxx, :])
        crop_num += 1
    elif dy >= 160 - epsy :
        dy1, dy2, dy3 = miny, miny + 1 * int(dy / 2), maxy
        save_pth = './crop40/cropimg_' + str(crop_num) + '.jpg'
        cv2.imwrite(save_pth, img_original[dy1: dy2, minx:maxx, :])
        crop_num += 1
        save_pth = './crop40/cropimg_' + str(crop_num) + '.jpg'
        cv2.imwrite(save_pth, img_original[dy2: dy3, minx:maxx, :])
        crop_num += 1
    elif dx <= 200 + epsx :
        dx1, dx2 = minx, maxx
        save_pth = './crop40/cropimg_' + str(crop_num) + '.jpg'
        cv2.imwrite(save_pth, img_original[miny:maxy, dx1:dx2, :])
        crop_num += 1
    else :
        pass

def hough(ori, img) :
    """
    对图片做霍夫变换，检测到最长边长，并且获得旋转角度，将旋转利用在原图中，得到校正后的图像。
    :param ori: 原始图像
    :param img: 二值化图像
    :return:
    返回旋转矫正之后的图像
    """
    lines = cv2.HoughLines(img, 1, np.pi/180, 50)
    result = ori.copy()
    # print("line shape is : ",lines.shape)
    for line in lines[0] :
        rho = line[0]
        theta = line[1]
        print(rho, theta, theta * 180 / np.pi)
        if (theta < (np.pi/4.0)) or (theta > (3.0 * np.pi/4.0)):
            print("Contidion 1")
            pt1 = (int(rho/np.cos(theta)),0)
            pt2 = (int((rho - result.shape[0] * np.sin(theta))/np.cos(theta)), result.shape[0])
            cv2.line(result, pt1, pt2,(0,0,255),2)
        else :
            print("Condition 2")
            pt1 = (0, int(rho / np.sin(theta)))
            pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta))/np.sin(theta)))
            cv2.line(result, pt1, pt2,(0,0,255), 2)
    # cv2.imshow("Result", result)
    # rotate_angle = 90 - theta * 180 / np.pi
    rotate_angle = theta * 180 / np.pi - 90
    h, w, _ = result.shape
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
    rotated = cv2.warpAffine(result, M, (w,h), flags = cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    #cv2.imshow("rotate", rotated)
    return rotated

def remove_plate_upanddown_border(card_img):
    """
    这个函数将截取到的车牌照片转化为灰度图，然后去除车牌的上下无用的边缘部分，确定上下边框
    输入： card_img是从原始图片中分割出的车牌照片
    输出: 在高度上缩小后的字符二值图片
    """
    #转灰度图像
    img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
 #   if np.sum(img)*255 < img.shape[0]*img.shape[1]*0.5
#    hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
    #img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
  #  hue, saturation, value = cv2.split(hsv)
  #  img = value
    #cv2.imshow("c",img)
    kernel = np.ones((2, 2), np.uint8)

    #开操作收缩窄缝隙，阈值分割
    plate_binary_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    plate_binary_img = cv2.morphologyEx(plate_binary_img, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("2",plate_binary_img)
    ret, plate_binary_img = cv2.threshold(plate_binary_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.imshow("3", plate_binary_img)
    #plate_binary_img = cv2.morphologyEx(plate_binary_img, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("4", plate_binary_img)
    #黑白转换，转成黑字
    height,width = plate_binary_img.shape
    print(height,width)
    if np.sum(plate_binary_img)<(height*width*255)*0.4:
        plate_binary_img = 255 - plate_binary_img
    #cv2.imshow("4", plate_binary_img)

    #求每一行最大的灰度值，不求导是因为求导会导致边界留下来，车牌边界有时候离文字很远
    row_histogram = np.sum(plate_binary_img, axis=1)  # 数组的每一行求和
    #row_min = np.min(row_histogram)
    #row_average = np.sum(row_histogram) / plate_binary_img.shape[0]
    #row_threshold = (row_min + row_average) / 2
    #wave_peaks = find_waves(row_threshold, row_histogram)
    # 接下来挑选最大的histogram
    his_up_max = 0.0
    his_down_max = 0.0
    up_location = 0
    down_location = 0
    #for row in row_histogram:
    for i, x in enumerate(row_histogram):

        if i < (len(row_histogram)*0.55):
            if x >= 0.93*his_up_max:#0.9很重要
                his_up_max = x
                up_location = i
        elif i >=(len(row_histogram)*0.6):#因为3号车牌中线太靠下了
            if x > his_down_max:
                # if his_down_max <= 0.95*x :
                his_down_max = x
                down_location = i
                if his_down_max >= 0.85 * width *255:
                    break
                # else:
                #     his_down_max = his_down_max
                #     down_location = down_location
            elif lastx <= his_down_max*0.65 and x> his_down_max:
                down_location = i
                break
        lastx = x
    plate_binary_img = plate_binary_img[up_location:down_location, :]
    #cv2.imshow("plate_binary_img_row", plate_binary_img)
    return plate_binary_img

def cut_img(img):
    """
    裁剪图像字符
    输入： img是处理过的车牌
    输出:  字符数组
    """
    cha_num = 0

    #求每一行最大的灰度值，不求导是因为求导会导致边界留下来，车牌边界有时候离文字很远
    col_histogram = np.sum(img, axis=0)  # 数组的每一列求和
    left_location = 0
    right_location = 0
    detect_left = 1
    height,width = img.shape
    partions = []
    print(height)
    last_his = 0
    #for row in row_histogram:
    for i, x in enumerate(col_histogram):
        if detect_left:
            if x <= 0.95 * height * 255 and last_his>= 0.95 * height * 255:
                left_location = i-1
                detect_left = 0 #调整为另一边
        else:
            if x > 0.9 * height * 255 :#and x < 0.4*height *255:
                right_location = i
                detect_left = 1
                if (right_location-left_location)>=10 :
#                    if (right_location-left_location)*height >14*38:
                        partions.append(img[:,left_location:right_location])
#                        save_pth = './character40/cropimg_' + str(crop_num) + "_"+str(cha_num) + '.jpg'
 #                       cv2.imwrite(save_pth, img[:, left_location:right_location])
                        cha_num += 1
                else:
                    detect_left=0
        last_his = x

    return partions

def judge(partion):
    """
    判断字符
    输入： 字符图片
    输出:  判断的值
    """

    path = './model/'

    row_histogram = np.sum(partion, axis=1)  # 数组的每一行求和
    his_max = partion.shape[1]*255
    up_location = 0
    for i, x in enumerate(row_histogram):
        if i < (len(row_histogram)/2):
            if x >= 0.9*his_max:#0.9很重要
                up_location = i
    partion = partion[up_location:, :]
    #print(partion.shape)
    partion = cv2.resize(partion, (20, 40))
    for dirpath, dirnames, filenames in os.walk(path):
        models = []
        for filename in filenames:
            img_path = os.path.join(dirpath, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #print(img.shape)
            models.append(img)
    names=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    partion = 255-partion

    max_distance = 0
    number = 0
    for i, model in enumerate(models):
        #四个边界为0
        model[0,:]=0
        model[39,:]=0
        model[:,0]=0
        model[:,19]=0
        result = cv2.matchTemplate(partion, model, cv2.TM_CCORR_NORMED)
        distance = np.max(result)
        if distance > max_distance:
            max_distance= distance
            number = i
    #print(max_distance)
    if max_distance > 0.4:
        print(names[number])

if __name__ == '__main__' :
    pth = 'License_plates.jpg'
    img = cv2.imread(pth)
    img = cv2.resize(img, (292 * 4, 173 * 4))
    cv2.imshow("original",img)
    # 1）提取单通道图片，选项为 （灰度图片/HSV中的value分支）
    singlechannel_img = SingleChannel(img)
    # 2）提升对比度
    contrast_img = Contrast(singlechannel_img)
    # contrast_img = singlechannel_img
    # 3）边缘连接（膨胀）
    kernel = np.ones((2, 2), np.uint8)
    dilation_img = cv2.dilate(contrast_img, kernel, iterations=1)
    cv2.imshow("dilate", dilation_img)
    # dilation_img = contrast_img
    # 4） 二值化
    threshold_img = threshold(dilation_img)
    # 5）利用findcontours函数找到边缘
    contoursList = drawCoutrous(threshold_img)
    # 6） 裁剪图片，车牌图片存储
    DrawRectangle(img, threshold_img, contoursList)
    for i in range(1):
        crop_num = i
        path = './crop40/cropimg_' + str(crop_num) + '.jpg'

        img = cv2.imread(path)
        cv2.imshow("img_test", img)
        # 7) 对车牌预处理
        contract = Contrast_2(img)
        threshold = threshold_2(contract)
        # 8）方向矫正
        hough_img = hough(img, threshold)
        # 9）车牌精确区域搜索
        plate_binary_img = remove_plate_upanddown_border(hough_img)
        # cv2.imshow(split_licensePlate_character(plate_binary_img))
        # 10） 字符分割
        partions = cut_img(plate_binary_img)
        for j, partion in enumerate(partions):
            cv2.imshow("partion"+str(j), partion)
        # 11） 字符识别
        judge(partion)
    cv2.waitKey()
    cv2.destroyAllWindows()

