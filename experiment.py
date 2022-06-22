import cv2
import numpy as np
from experiment import motion, optical_flow
from utils import image as img_utils


if __name__ == '__main__':
    folder_path = "dataset/data_151_221/data (151)"

    # optical_flow.display_optical_flow(folder_path)

    imgs = img_utils.read_from_folder(folder_path)

    # step = 5
    # diff_list = []
    # for i in range(0, len(imgs) - step, step):
    #     diff_list.append(motion.diff(imgs[i], imgs[i + step]))

    length = 75
    motion_list = []
    # for i in range(len(imgs) - length):
    #     motion_list.append(motion.motion_blur(imgs[i:i + length]))

    motion_list = motion.optical_flow(imgs)

    img_utils.play_img_seq(motion_list, 30)
