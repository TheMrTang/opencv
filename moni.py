import numpy as np
import cv2 as cv2
import math
import serial
import struct

COLOR = (255, 50, 200)
com = serial.Serial("/dev/ttyTHS1", 115200)

def imread_photo(filename, flags = cv2.IMREAD_COLOR):

    """
    该函数能够读取磁盘中的图片文件，默认以彩色图像的方式进行读取
    输入： filename 指的图像文件名（可以包括路径）
          flags用来表示按照什么方式读取图片,有以下选择（默认采用彩色图像的方式）：
              IMREAD_COLOR 彩色图像
              IMREAD_GRAYSCALE 灰度图像
              IMREAD_ANYCOLOR 任意图像
    输出: 返回图片的通道矩阵
    """
    return cv2.imread(filename, flags)

def resize_photo(imgArr, MAX_WIDTH = 800):

    """
    这个函数的作用就是来调整图像的尺寸大小，当输入图像尺寸的宽度大于阈值（默认1000），我们会将图像按比例缩小
    输入： imgArr是输入的图像数字矩阵
    输出:  经过调整后的图像数字矩阵
    拓展：OpenCV自带的cv2.resize()函数可以实现放大与缩小，函数声明如下：
            cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) → dst
        其参数解释如下：
            src 输入图像矩阵
            dsize 二元元祖（宽，高），即输出图像的大小
            dst 输出图像矩阵
            fx 在水平方向上缩放比例，默认值为0
            fy 在垂直方向上缩放比例，默认值为0
            interpolation 插值法，如INTER_NEAREST，INTER_LINEAR，INTER_AREA，INTER_CUBIC，INTER_LANCZOS4等            
    """
    img = imgArr
    rows, clos = img.shape[:2]
    if clos > MAX_WIDTH:
        change_rate = MAX_WIDTH / clos
        img = cv2.resize(img, (MAX_WIDTH, int(rows * change_rate)), interpolation=cv2.INTER_AREA)
    return img

def predict(imgArr):
    """
        通过一系列处理找到矩形区域，并返回面积最大的矩形的坐标及其近似轮廓坐标点。

        输入:
        - imgArr: 原始图像的数字矩阵

        输出:
        - max_area_rectangle: 面积最大的矩形的坐标，格式为 (x, y, w, h)，其中 (x, y) 是左上角坐标，w 是宽度，h 是高度。
        - max_approx: 面积最大的矩形的近似轮廓坐标点。
    """
    # 设置Canny边缘检测的参数
    minValue = 50
    maxValue = 150
    # 存储找到的矩形坐标
    rectangles = []
    # 记录面积最大的矩形信息
    max_area = 0
    max_area_rectangle = None
    max_approx = None
    # 复制原始图像
    img_copy = imgArr.copy()
    # Canny边缘检测
    img_canny = cv2.Canny(img_copy, minValue, maxValue)
    # 轮廓检测
    contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历轮廓
    for contour in contours:
        # 计算轮廓的周长和使用多边形逼近函数得到近似轮廓
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

         # 如果近似轮廓是矩形（拥有4个顶点），则记录其矩形边框坐标
        if len(approx)==4:
            x, y, w, h = cv2.boundingRect(approx)
            rectangles.append((x, y, w, h))

            # 计算轮廓的面积
            area = cv2.contourArea(contour)

            # 更新记录面积最大的矩形的信息
            if area > max_area:
                max_area = area
                max_area_rectangle = (x, y, w, h)
                max_approx = approx

    return max_area_rectangle, max_approx

def draw_line(img, mode, x, y, rat):
    """
    该函数能根据mode的值，找出距离矩形左上角坐标20/30/40cm的坐标，并以坐上角坐标为起点，找到的坐标为终点，画一条线段。
    输入：
        img：图像
        mode：控制绘画的直线长度
        x：矩形左上角坐标的x值
        y：矩形左上角坐标的y值
        rat：cm和像素的转换比例
    输出：
        最终坐标
    """
    if mode == 1:
        lenth = 20 / rat
    elif mode == 2:
        lenth = 30 / rat
    elif mode ==3:
        lenth = 40 / rat

    point_x = x + int(0.5 * lenth)
    point_y = y + int(0.866 * lenth)
    
    cv2.line(img, (x, y), (point_x, point_y), COLOR, 2)
    packed_data = struct.pack("<4B", 0x2C, 0x12, point_x, point_y)
    com.write(packed_data)
    return (point_x, point_y)

def draw_rect(img, mode, center_x, center_y, rat):
    """
    该函数是用于绘画20/30/40cm边长的正方形
    输入：
        img:图像
        mode:控制边长
        center_x:屏幕中心点坐标的x值
        center_y:屏幕中心点坐标的y值
        rat:cm和像素的转换比例
    输出：
        正方形的顺时针坐标点
    """
    points = []
    if mode == 1:
        lenth = 20 / rat
    elif mode == 2:
        lenth = 30 / rat
    elif mode ==3:
        lenth = 40 / rat
    left_up_x = center_x - int(lenth / 2)
    left_up_y = center_y - int(lenth / 2)
    right_up_x = center_x + int(lenth / 2)
    right_up_y = center_y - int(lenth / 2)
    right_down_x = center_x + int(lenth / 2)
    right_down_y = center_y + int(lenth / 2)
    left_down_x = center_x - int(lenth / 2)
    left_down_y = center_y + int(lenth / 2)
    points.append(left_up_x)
    points.append(left_up_y)
    points.append(right_up_x)
    points.append(right_up_y)
    points.append(right_down_x)
    points.append(right_down_y)
    points.append(left_down_x)
    points.append(left_down_y)
    cv2.rectangle(img, (left_up_x, left_up_y), (right_down_x, right_down_y), COLOR, 2)
    packed_data = struct.pack('<10B', 0x2C, 0x12, *points)
    com.write(packed_data)
    return [(left_up_x, left_up_y), (right_up_x, right_up_y), (right_down_x, right_down_y), (left_down_x, left_down_y)]

def draw_circle(img, mode, center_x, center_y, rat):
    """
    该函数用于画圆半径为10/20cm
    输入：
        img:图像
        mode:控制圆的半径
        center_x:屏幕中心点坐标的x值
        center_y:屏幕中心点坐标的y值
        rat:cm和像素的转换比例
    输出：
        圆环上的点
    """
    PI = 3.141592657
    num = 50
    points = []
    alpha = 2 * PI / (num - 1)  # 计算角度的增量，以便在圆周上均匀取点。

    if mode == 1:
        R = 10 / rat
    elif mode == 2:
        R = 20 / rat
    
    # 计算圆上的点的坐标，并将它们添加到 points 列表中
    for i in range(num):
        point_x = center_x + int(R * math.cos(i * alpha))
        point_y = center_y + int(R * math.sin(i * alpha))
        points.append((point_x, point_y))
    
    # 在图像上绘制连接圆上相邻点的线段
    for i in range(num-1):
        i += 1
        cv2.line(img, points[i-1], points[i], COLOR, 2)

    packed_Data = struct.pack('<102B', 0x2C, 0x12, *points)
    print(packed_Data)
    com.write(packed_Data)

    return points

#shi-Tomas算法
'''
shi-Tomas算法是对Harris角点检测算法的改进,一般会比Harris算法得到更好的角点.
Harris算法的角点响应函数是将矩阵M的行列式值与M的迹相减,利用差值判断是否为角点,
后来shi和Tomas提出改进的方法是,若矩阵M的两个特征值中较小的一个大于阈值,则认为
他是角点,即:
    R=min(a1,a2)

API:
    corners=cv.goodFeaturesToTrack(image,maxcorners,qualityLevel,minDistance)
    参数:
        image:输入的灰度图像
        maxCorners:获取角点数的数目
        qualityLevel:该参数指出最低可接受的角点质量水平,在0~1之间
        minDistance:角点之间的最小欧氏距离,避免得到相邻特征点
    返回:
        corners:搜索到的角点,在这里所有低于质量水平的角点被排除,然后把合格的角点按照质量排序,
        然后将质量较好的角点附近(小于最小欧氏距离)的角点删除,最后找到maxCorners个角点返回

'''
if __name__ == "__main__":
    img = cv2.imread('E:\PyCharm\Project\OpenCv\sz1\\2.jpg')
    img = resize_photo(img)
    points, approx = predict(img)
    # 矩形左上角坐标
    x = points[0]
    y = points[1]
    # 中心点坐标
    center_x = points[0] + (points[2] // 2)
    center_y = points[1] + (points[3] // 2)
    # 矩形的宽、高
    w = points[2]
    h = points[3]

    # 计算比例系数
    lenth_pc = cv2.norm(np.array((x, y)) - np.array((center_x, center_y)))
    lenth = 30 * 1.41421
    rat = lenth / lenth_pc

    cv2.drawContours(img, [approx], -1, COLOR, 2)
    cv2.circle(img, (center_x, center_y),  2, COLOR, -1)
    print(draw_line(img, 2, x, y, rat))
    print(draw_rect(img, 1, center_x, center_y, rat))
    print(draw_circle(img, 2, center_x, center_y, rat))
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# import cv2
# class tup:
    
#     def __init__(self,dizhi):
#         self.__tupian = cv2.imread(dizhi)
#         self.__tupian = cv2.resize(self.__tupian,(640,480))
#         self.__erzhi = cv2.cvtColor(self.__tupian,cv2.COLOR_BGR2GRAY)
        
#     def xianshi_yuanshi(self):
#         cv2.imshow("cv1",self.__tupian)
#         cv2.imshow("cv2",self.edges)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     def Xianshi(self):
#         cv2.imshow("cv2",self.__erzhi)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     def find_contour(self):
#         blurred = cv2.GaussianBlur(self.__erzhi, (5, 5), 0)

# # 使用Canny边缘检测找到图像的边缘
#         self.edges = cv2.Canny(blurred, 50, 150)

#         # 找到图像中的轮廓
#         contours, _ = cv2.findContours(self.edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#         # 遍历所有轮廓
#         for contour in contours:
#             # 计算轮廓的周长
#             perimeter = cv2.arcLength(contour, True)

#             # 使用approxPolyDP函数近似找到多边形
#             epsilon = 0.01 * perimeter
#             approx = cv2.approxPolyDP(contour, epsilon, True)

#             # 如果找到的多边形是四边形，认为是矩形
#             if len(approx) == 4:
#                 cv2.drawContours(self.__tupian, [approx], 0, (0, 255, 0), 2)

# if __name__=="__main__":
#     t1 = tup('E:\PyCharm\Project\OpenCv\sz1\\1.jpg')
#     t1.find_contour()
#     t1.xianshi_yuanshi()