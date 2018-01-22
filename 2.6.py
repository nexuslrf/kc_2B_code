# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 20:25:00 2014
@author: duan
"""
import cv2
import numpy as np
def nothing(x):
    pass
cap=cv2.VideoCapture(1)
cv2.namedWindow('frame')
cv2.createTrackbar('low', 'frame', 0, 179, nothing)
cv2.createTrackbar('up', 'frame', 0, 179, nothing)
while(1):
    # 获取每一帧
    ret,frame=cap.read()
    # 转换到HSV
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # 设定蓝色的阈值
    l=cv2.getTrackbarPos('low','frame')
    u=cv2.getTrackbarPos('up','frame')
    lower_blue=np.array([l,50,50])
    upper_blue=np.array([u,255,255])
    # 根据阈值构建掩模
    mask=cv2.inRange(hsv,lower_blue,upper_blue)
    # 对原图像和掩模进行位运算
    res=cv2.bitwise_and(frame,frame,mask=mask)
    # 显示图像
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k=cv2.waitKey(1)&0xFF
    if k==ord('q'):
        break
# 关闭窗口
cv2.destroyAllWindows()