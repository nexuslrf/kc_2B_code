import cv2
# OpenCV 人脸检测
face_patterns = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
sample_image = cv2.imread('img/face.jpg')
faces = face_patterns.detectMultiScale(sample_image, scaleFactor=1.1, minNeighbors=8, minSize=(50, 50))


