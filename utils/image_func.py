from natsort import natsorted
import cv2
import glob
import numpy as np
import os
import sys


def imread_list(file_name: str): 
    img_list = [cv2.imread(file) for file in natsorted(glob.glob(file_name + "/*.jpg"))]
    return img_list

def img_diff(img_a: np.ndarray, img_b:np.ndarray): 
    diff = abs(img_a.astype(np.float32) - img_b.astype(np.float32))
    return diff.astype(np.uint8)

def motion(img_list:list, img_a: int, img_b:int): 
    motion_img = sum(img_list[img_a : img_b])
    return motion_img

def resize(filename: str, new_size: int, channels: int = 3):
    file_path = os.path.join(sys.path[0], filename)
    img = cv2.imread(file_path)

    h, w = img.shape[:2]
    img_size = max(h, w)
    img_new = np.zeros((img_size, img_size, channels)).astype(np.uint8)
    img_new[(w-h)//2:w-(w-h)//2, 0:w] = img
    img_new = cv2.resize(img_new, (new_size, new_size))

    return img_new


if __name__ == '__main__':
    for j in range(1, 25):
        for k in range(1,9):
            diff_list = []
            input = "./FDD_data_picture/data ("  + str(j) + "_" + str(k) + ")"
            output_path = "./motion/data ("  + str(j) + "_" + str(k) + ")/"
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            images = imread_list(input)
            for i in range(len(images) - 1): 
                diff_list.append(img_diff(images[i], images[i + 1]))
                if i >= 4: 
                    img = motion(diff_list, i - 4, i)
                    cv2.imwrite(output_path + str(i) + ".jpg", img)
