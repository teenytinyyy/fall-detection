import cv2
import numpy as np
from experiment import motion
from utils import image as img_utils
import os


if __name__ == '__main__':

    folder_path = "dataset/data/FDD_data_picture/data (1_1)"
    output_path = "dataset/data/MHI/data (1_1)"
    if not os.path.isdir(output_path):
                os.makedirs(output_path)

    imgs = img_utils.read_from_folder(folder_path)

    length = 21
    motion_list = []

    # XXX display input video
    # motion_list = imgs

    # XXX display motion blur
    # for i in range(len(imgs) - length):
    #     motion_list.append(motion.motion_blur(imgs[i:i + length]))

    # XXX display MEI
    # for i in range(len(imgs) - length):
    #     motion_list.append(motion.motion_energy_image(imgs[i:i + length]))

    # XXX display MHI
    for i in range(len(imgs) - length):
        motion_list.append(motion.motion_history_image(imgs[i:i + length]))

    # XXX display optical flow
    # motion_list = motion.optical_flow(imgs)

    img_utils.write_img_seq(motion_list, output_path)
