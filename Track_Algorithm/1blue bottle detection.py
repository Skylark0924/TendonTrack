#追踪蓝色瓶子，并且显示外接圆+画出重心#
# 导入所需模块
import cv2 as cv
import numpy as np
import imutils

# 打开摄像头
cap = cv.VideoCapture(0)

while True:
    # 读取每一帧
    _, frame = cap.read()
    # 重设图片尺寸以提高计算速度
    if frame is not False:
        frame = imutils.resize(frame, width=600)
        # 进行高斯模糊
        blurred = cv.GaussianBlur(frame, (11, 11), 0)
        # 转换颜色空间到HSV
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        # 定义红色无图的HSV阈值
        lower_red = np.array([110, 50, 50])
        upper_red = np.array([130, 255, 255])
        # 对图片进行二值化处理
        mask = cv.inRange(hsv, lower_red, upper_red)
        # 腐蚀操作
        mask = cv.erode(mask, None, iterations=2)
        # 膨胀操作，先腐蚀后膨胀以滤除噪声
        mask = cv.dilate(mask, None, iterations=2)
        cv.imshow('mask', mask)
        # 寻找图中轮廓
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
        # 如果存在至少一个轮廓则进行如下操作
        if len(cnts) > 0:
            # 找到面积最大的轮廓
            c = max(cnts, key=cv.contourArea)
            # 使用最小外接圆圈出面积最大的轮廓
            ((x, y), radius) = cv.minEnclosingCircle(c)
            # 计算轮廓的矩
            M = cv.moments(c)
            # 计算轮廓的重心
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # 只处理尺寸足够大的轮廓
            if radius > 5:
                # 画出最小外接圆
                cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                # 画出重心
                cv.circle(frame, center, 5, (0, 0, 255), -1)
                font = cv.FONT_HERSHEY_SIMPLEX
                text = 'bottle'
                cv.putText(frame, text, (212, 310), font, 1, (0, 0, 255), 3)
        cv.imshow('frame', frame)
        k = cv.waitKey(5) & 0xFF
        if k == ord('q'):
            break
cap.release()
cv.destroyAllWindows()

'''
>>> green = np.uint8([[[0,255,0 ]]])
>>> hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
>>> print( hsv_green )
[[[ 60 255 255]]]
现在分别将[H-10,100,100]和[H + 10,255,255]作为下限和上限。除了这种方法，
您可以使用任何图像编辑工具（如GIMP或任何在线转换器）来查找这些值，但不要忘记调整HSV范围。
'''

# 彩色模型
# 数字图像处理中常用的采用模型是RGB（红，绿，蓝）模型和HSV（色调，饱和度，亮度），RGB广泛应用于彩色监视器和彩色视频摄像机，我们平时的图片一般都是RGB模型。而HSV模型更符合人描述和解释颜色的方式，HSV的彩色描述对人来说是自然且非常直观的。
#
# HSV模型
# HSV模型中颜色的参数分别是：色调（H：hue），饱和度（S：saturation），亮度（V：value）。由A. R. Smith在1978年创建的一种颜色空间, 也称六角锥体模型(Hexcone Model)。
#
# 色调（H：hue）：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°。它们的补色是：黄色为60°，青色为180°,品红为300°；
# 饱和度（S：saturation）：取值范围为0.0～1.0，值越大，颜色越饱和。
# 亮度（V：value）：取值范围为0(黑色)～255(白色)。
# OpenCV下有个函数可以直接将RGB模型转换为HSV模型，注意的是OpenCV中H∈ [0, 180）， S ∈ [0, 255]， V ∈ [0, 255]。我们知道H分量基本能表示一个物体的颜色，但是S和V的取值也要在一定范围内，因为S代表的是H所表示的那个颜色和白色的混合程度，也就说S越小，颜色越发白，也就是越浅；V代表的是H所表示的那个颜色和黑色的混合程度，也就说V越小，颜色越发黑。经过实验，识别蓝色的取值是 H在100到140，S和V都在90到255之间。一些基本的颜色H的取值可以如下设置：
#常见颜色的H取值
# Orange  0-22
# Yellow 22-38
# Green 38-75
# Blue 75-130
# Violet 130-160
# Red 160-179

