import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('origin.png')
dst=cv2.pyrMeanShiftFiltering(img, 25, 10)
cv2.imshow("img",img)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
