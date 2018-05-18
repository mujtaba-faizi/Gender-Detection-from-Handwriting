import cv2
import numpy as np

image = cv2.imread("Datasets/Big/0001_1.jpg",0)
ret, thresh1 = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)
thresh1 = cv2.medianBlur(thresh1, 7)
array=np.asarray(thresh1)
array=np.reshape(array, array.shape + (1,))
cv2.imwrite("d.jpg",array)
image = cv2.imread("d.jpg")
array=np.asarray(image)
print(array)

