# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 20:25:00 2014
@author: duan
"""
import cv2
import numpy as np
import time
cap=cv2.VideoCapture(2)
flag=True
def nothing(x):
    pass
cv2.namedWindow('res')
cv2.createTrackbar('Hmin','res',0,180,nothing)
cv2.createTrackbar('Smin','res',0,255,nothing)
cv2.createTrackbar('Vmin','res',0,255,nothing)
cv2.createTrackbar('Hmax','res',0,270,nothing)
cv2.createTrackbar('Smax','res',0,255,nothing)
cv2.createTrackbar('Vmax','res',0,255,nothing)
while(1):
    # 获取每一帧
    Hmin=cv2.getTrackbarPos('Hmin','res')
    Smin=cv2.getTrackbarPos('Smin','res')
    Vmin=cv2.getTrackbarPos('Vmin','res')
    Hmax=cv2.getTrackbarPos('Hmax','res')
    Smax=cv2.getTrackbarPos('Hmax','res')
    Vmax=cv2.getTrackbarPos('Hmax','res')
    ret,frame=cap.read()
    frame = cv2.GaussianBlur(frame, (33, 33), 0)

    # 转换到HSV
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # 设定绿色的阈值
    lower_green = np.array([63, 100, 70])
    upper_green = np.array([86, 255, 255])
    #黄色
    # lower_yellow=np.array([24,100,100])
    # upper_yellow=np.array([32,255,255])
    # 红色
    lower_red=np.array([165, 80, 100])
    upper_red=np.array([180, 255, 255])
    # lower_red = np.array([Hmin, Smin, Vmin])
    # upper_red = np.array([Hmax, Smax, Vmax])
    # 根据阈值构建掩模
    mask_red=cv2.inRange(hsv, lower_red, upper_red)
    mask_green=cv2.inRange(hsv,lower_green,upper_green)
    # 对原图像和掩模进行位运算
    res=cv2.bitwise_and(frame, frame, mask=mask_green + mask_red)
    # 显示图像
    image, contours_green, hierarchy = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image, contours_red, hierarchy = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(frame, contours_green, 0, (0, 255, 0), 1)
    img = cv2.drawContours(img, contours_red, 0, (0, 255, 0), 1)
    cv2.imshow('frame',img)
    #cv2.imshow('mask_green',mask_green)
    cx_green=0
    cy_green=0
    cx_red=0
    cy_red=0
    if len(contours_green):
        cnt = contours_green[0]
        M_green = cv2.moments(cnt)
        if M_green['m00']!=0:
            cx_green = int(M_green['m10'] / M_green['m00'])
            cy_green = int(M_green['m01'] / M_green['m00'])
    if len(contours_red):
        cnt = contours_red[0]
        M_red = cv2.moments(cnt)
        if M_red['m00']!=0:
            cx_red = int(M_red['m10'] / M_red['m00'])
            cy_red = int(M_red['m01'] / M_red['m00'])
    cx= (cx_green + cx_red) / 2
    cy= (cy_green + cy_red) / 2
    print cx,cy
    cv2.circle(res,(cx,cy),20,(255,0,0),-1)
    cv2.imshow('res', res)
    k=cv2.waitKey(5)&0xFF
    if k==ord('q'):
        break
    time.sleep(0.01)
# 关闭窗口
cv2.destroyAllWindows()