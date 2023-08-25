import cv2
import numpy as np
from matplotlib import pyplot as plt


def seg_bg_fg(img):
    # 基础处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 2)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 简单阈值分割(自动获取阈值)
    # cv2.imshow('thresh', thresh)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  # 开运算降噪 #thresh
    cv2.imshow('opening', opening)
    conv_bg = cv2.dilate(opening, kernel, iterations=3)  # 膨胀得到确定的背景
    # cv2.imshow('conv_bg', conv_bg)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # tmp = dist_transform.max()
    # print(tmp)
    # img[dist_transform == tmp] = [0, 255, 0]
    ret, conv_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)  # 以到中心点的距离为阈值得到确定的前景
    dist_transform = cv2.convertScaleAbs(dist_transform)
    cv2.imshow('dist_transform', dist_transform)
    # cv2.imshow('conv_fg', conv_fg)
    conv_fg = np.uint8(conv_fg)
    unknown = cv2.subtract(conv_bg, conv_fg)  # 做差,留下未知区域 # conv_bg,
    # cv2.imshow('unknown', unknown)

    # 应用分水岭,对每个区域打标记
    ret, markers = cv2.connectedComponents(conv_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    # cv2.imshow("show", markers)
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    img_marker = cv2.applyColorMap(cv2.convertScaleAbs(markers, alpha=15), cv2.COLORMAP_JET)
    img_mix = cv2.addWeighted(img, 0.8, img_marker, 0.2, 0)

    cv2.imshow('marker', img_marker)
    cv2.imshow('img', img_mix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_cent(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # 简单阈值分割(自动获取阈值)
    # cv2.imshow("thresh", thresh)
    cnts = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    for c in cnts:
        # 获取中心点
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # 画出轮廓和中点
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(img, "center", (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 显示图像
        cv2.imshow("Image", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_contor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3, 3), 2)
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 简单阈值分割(自动获取阈值)
    # cv2.imshow('thresh', thresh)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)  # 开运算降噪 #thresh
    # cv2.imshow('opening', opening)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    c = cv2.dilate(opening, k)
    d = cv2.erode(opening, k)
    dst = cv2.subtract(c, d)
    '''
    cv2.imshow('img', c)
    cv2.waitKey(0)
    cv2.imshow('img', d)
    cv2.waitKey(0)
    cv2.imshow('img', dst)
    cv2.waitKey(0)
    '''
    # thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # 简单阈值分割(自动获取阈值)
    thresh = cv2.threshold(dst, max(0.5 * dst.max(), 100), 255, cv2.THRESH_BINARY_INV)[1]
    thresh = cv2.bitwise_not(thresh)
    # cv2.imshow('thresh', thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # circles = cv2.HoughCircles(opening.copy(), cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    if contours is None:
        return (None, None), img

    # cnts = np.uint16(np.around(circles))
    # cnts = cnts[0]

    circle_cnt = 0
    cXY = []
    for c in contours:
        # 轮廓逼近
        epsilon = 0.005 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        # 分析几何形状
        corners = len(approx)
        if corners < 15:
            # print("Not circle")
            continue
        else:
            # print("Circle")
            circle_cnt += 1

        # 获取中心点
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX_tmp = int((M["m10"] + 1e-10) / (M["m00"] + 1e-10))
        cY_tmp = int((M["m01"] + 1e-10) / (M["m00"] + 1e-10))
        # print(cX_tmp, cY_tmp)
        # 画出轮廓和中点
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        cv2.circle(img, (cX_tmp, cY_tmp), 7, (255, 255, 255), -1)
        cv2.putText(img, "center", (cX_tmp - 20, cY_tmp - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cXY.append([cX_tmp, cY_tmp])
        # 显示图像
        # cv2.imshow("Image", img)
        # cv2.waitKey(2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if circle_cnt >= 1:
        return cXY, img
    else:
        return [], img


'''
    for i in cnts:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(img, (i[0], i[1]), 2, (255, 255, 255), 2)
        cv2.putText(img, "center", (i[0] - 20, i[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        # cv2.imshow("Image", img)
        # cv2.waitKey(2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return (i[0], i[1]), img
'''

if __name__ == "__main__":
    img = cv2.imread('./tmp.jpg')  # /home/lab/Github/TendonTrack/Simulator/utils/image_log
    # seg_bg_fg(img)
    detect_contor(img)
