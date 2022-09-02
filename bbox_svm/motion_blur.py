from natsort import natsorted
import cv2
import glob
import numpy as np
import os


def imread_list(file_name: list):
    img_list = [cv2.imread(file)
                for file in natsorted(glob.glob(file_name + "/*.jpg"))]
    return img_list


def img_diff(img_a: np.ndarray, img_b: np.ndarray):
    diff = abs(img_a.astype(np.float32) - img_b.astype(np.float32))
    return diff.astype(np.uint8)


def motion(img_list: list, img_a_idx: int, img_b_idx: int, thres: int):
    motion_img = sum(img_list[img_a_idx: img_b_idx: 2])
    motion_img[motion_img < thres] = 0
    return motion_img


if __name__ == '__main__':
    for j in range(1, 2):
        for k in range(1,9):
            diff_list = []
            input = "../dataset/data/FDD_data_picture/data ("  + str(j) + "_" + str(k) + ")"
            output_path = "../dataset/data/motion_3/data ("  + str(j) + "_" + str(k) + ")/"
            #input = "./FDD_data_picture/data (" + str(j) + ")"
            #output_path = "./motion_9/dataset_15/data_" + str(j) + "/"
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            images = imread_list(input)
            for i in range(len(images) - 1):
                diff_list.append(img_diff(images[i], images[i + 1]))
                if i >= 3:
                    img = motion(diff_list, i - 3, i, 0)
                    cv2.imwrite(output_path + str(i) + ".jpg", img)
