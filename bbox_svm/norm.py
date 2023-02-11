import cv2

img = cv2.imread("../dataset/data/test1.jpg")
img_t, th = cv2.threshold(img, 150, 255, cv2.THRESH_TOZERO_INV)
img_normalized = cv2.normalize(th, None, 50, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
cv2.imwrite("../dataset/data/test1_norm.jpg", img_normalized)