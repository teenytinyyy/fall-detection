import cv2
import numpy as np
import os
import sys

def resize(img: np.ndarray, new_size: int, channels: int = 3):
    # file_path = os.path.join(sys.path[0], filename)
    # img = cv2.imread(file_path)

    h, w = img.shape[:2]
    img_size = max(h, w)
    img_new = np.zeros((img_size, img_size, channels)).astype(np.uint8)

    img_new[(w-h)//2:w-(w-h)//2, 0:w] = img
    img_new = cv2.resize(img_new, (new_size, new_size))
    return img_new