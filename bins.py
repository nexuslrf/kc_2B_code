import cv2
img=cv2.imread("messi.jpg")
print img.shape[0]
gray_img=cv2.imread("messi.jpg",cv2.IMREAD_GRAYSCALE)
print gray_img.shape

cv2.imwrite("messi2.jpg",gray_img)