import cv2
import numpy as np
import time
import math
import serial
# import struct

COLOR = (255, 50, 200)
cX1, cY1, cX2, cY2 = None, None, None, None
rat = None
center_x = None
center_y = None
PI = 3.141592657
com = serial.Serial("/dev/ttyTHS1",115200)

def recv(com):
	while 1:
		if com.in_waiting > 0:
			data = []
			data = com.read(10)
			if data == '' :
				continue
			else:
				break
		return data

def send_data(com, data):
	com.write(data)

def findLaser(img):
	"""
	找到图片中点绿色激光点与红色激光点并定位中心
	:param img: 需要处理点图片
	:return: 绿色激光点中心（x1, y1）;红色激光点中心（x2, y2)
	"""
	cX1, cY1, cX2, cY2 = None, None, None, None
	greenLaser = 'green'
	redLaser = 'red'
	# 色系下限上限表
	color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
		  'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
		  }
	# 灰度图像处理
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图
	# cv2.imshow('gray', gray)

	# 高斯滤波
	blurred = cv2.GaussianBlur(img, (11, 11), 0)
	# 创建运算核
	kernel = np.ones((1, 1), np.uint8)
	# 开运算
	opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
	# 二值化处理
	thresh = cv2.threshold(opening, 230, 255, cv2.THRESH_BINARY)[1]
	# cv2.imshow('thresh', thresh)
	# 转化成HSV图像
	hsv = cv2.cvtColor(thresh, cv2.COLOR_BGR2HSV)
	# 颜色二值化筛选处理
	inRange_hsv_green = cv2.inRange(hsv, color_dist[greenLaser]['Lower'], color_dist[greenLaser]['Upper'])
	# inRange_hsv_red = cv2.inRange(hsv, color_dist[redLaser]['Lower'], color_dist[redLaser]['Upper'])
	# cv2.imshow('inrange_hsv_green', inRange_hsv_green)
	# cv2.imshow('inrange_hsv_red', inRange_hsv_red)
	# 找绿色激光点+
	try:
		cnts1 = cv2.findContours(inRange_hsv_green.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		c1 = max(cnts1, key=cv2.contourArea)
		M = cv2.moments(c1)
		cX1 = int(M["m10"] / M["m00"])
		cY1 = int(M["m01"] / M["m00"])
		cv2.circle(img, (cX1, cY1), 3, (0, 255, 0), -1)
		# rect = cv2.minAreaRect(c1)
		# box = cv2.boxPoints(rect)
		# cv2.drawContours(img, [np.int0(box)], -1, (0, 255, 0), 2)
	except:
		print('没有找到绿色的激光')

	# 找红色激光点
	# try:
	#     cnts2 = cv2.findContours(inRange_hsv_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	#     c2 = max(cnts2, key=cv2.contourArea)
	#     M = cv2.moments(c2)
	#     cX2 = int(M["m10"] / M["m00"])
	#     cY2 = int(M["m01"] / M["m00"])
	#     cv2.circle(img, (cX2, cY2), 3, (0, 0, 255), -1)
	#     rect = cv2.minAreaRect(c2)
	#     box = cv2.boxPoints(rect)
	#     cv2.drawContours(img, [np.int0(box)], -1, (0, 0, 255), 2)
	# except:
	#     print('没有找到红色的激光')
	return cX1, cY1, cX2, cY2

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

def get_jiguang(img):
	global cX1, cY1
	global cX2, cY2

	if cX2 == None :
		if cX1 != None:
			cX2, cY2, _, _ = findLaser(img)
			return [(cX1, cY1), (cX2, cY2), 2]
	
	if cX1 == None :
		cX1, cY1, _, _ = findLaser(img)
		return [(cX1, cY1), (cX2, cY2), 1]

def find_contour(img):
	get_jiguang(img)
	if cX1 == None or cX2 == None:
		time.sleep(0.2)
	return [cX1, cY1, cX2, cY2]

def Calculated_constant():
	global cX1, cY1
	global cX2, cY2
	global rat
	global center_x
	global center_y

	if cX1 != None and cX2 != None :
		# 计算比例系数
		Constant = 50 * 1.41421
		lenth_pc = cv2.norm(np.array((cX1, cY1)) - np.array((cX2, cY2)))
		rat = Constant / lenth_pc
		# 计算中心坐标
		center_x = (cX1 + cX2) // 2
		center_y = (cY1 + cY2) // 2

def draw_line(img, lenth, x, y, rat, angle):
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
	points = []
	lenth = lenth / rat
	Rad = 2 * PI / 360 * angle
	point_x = x + int(math.sin(Rad) * lenth)
	point_y = y + int(math.cos(Rad) * lenth)
	points.append((point_x, point_y))
	cv2.line(img, (x, y), (point_x, point_y), COLOR, 2)
	# packed_data = pickle.dumps(points)
	# packed_data = struct.pack("<2B2h", 0x2C, 0x12, point_x, point_y)
	# com.write(0x2c+0x12+packed_data)

	return [point_x, point_y]

def draw_rect(img, lenth, center_x, center_y, rat):
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
	lenth = lenth / rat
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

	# packed_data = pickle.dumps(points)

	cv2.rectangle(img, (left_up_x, left_up_y), (right_down_x, right_down_y), COLOR, 2)
	# packed_data = struct.pack('<2B8h', 0x2C, 0x12, *points)
	# com.write(0x2c+0x12+packed_data)
	return points

def draw_circle(img, radius, center_x, center_y, rat):
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

	R = radius / rat
	
	# 计算圆上的点的坐标，并将它们添加到 points 列表中
	for i in range(num):
		point_x = center_x + int(R * math.cos(i * alpha))
		point_y = center_y + int(R * math.sin(i * alpha))
		points.append(point_x)
		points.append(point_y)
	
	# 在图像上绘制连接圆上相邻点的线段
	for i in range(2*num-1):
		i += 1
		cv2.line(img, points[i-1], points[i], COLOR, 2)
	
	# packed_Data = pickle.dumps(points)
	# com.write(0x2c+0x12+packed_Data)

	return points


if __name__ == '__main__':
	cap = cv2.VideoCapture(0)
	while cap.isOpened():
		_, frame = cap.read()
		img = resize_photo(frame)
		if recv(com) == 0x01:
			data = find_contour(img)
			for i in range(len(data)):
				send_data(com, data[i])
			Calculated_constant()
			# cv2.circle(img, (center_x, center_y),  2, COLOR, -1)
		elif recv(com) == 0x02:
			data = draw_line(img, 20, cX1, cY1, rat, 30)
			for i in range(len(data)):
				send_data(com, data[i])
		elif recv(com) == 0x03:
			data = draw_rect(img, 20, center_x, center_y, rat)
			for i in range(len(data)):
				send_data(com, data[i])
		elif recv(com) == 0x04:
			data = draw_circle(img, 10, center_x, center_y, rat)
			for i in range(len(data)):
				send_data(com, data[i])
		cv2.imshow("img", img)
		c = cv2.waitKey(1)
		if c == 27:
			break
	cap.release()
	cv2.destroyAllWindows()