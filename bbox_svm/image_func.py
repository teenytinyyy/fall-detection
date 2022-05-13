from natsort import natsorted
import cv2
import glob
import numpy as np

def imread_list(file_name:list): 
    img_list = [cv2.imread(file) for file in natsorted(glob.glob(file_name + "/*.jpg"))]
    return img_list

def img_diff(img_a: np.ndarray, img_b:np.ndarray): 
    diff = abs(img_a - img_b)
    return diff

def motion(img_list:list, img_a: np.ndarray, img_b:np.ndarray): 
    motion_img = sum(img_list[img_a.astype(np.float32) : img_b.astype(np.float32)])
    return motion_img.astype(np.uint8)