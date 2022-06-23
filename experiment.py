import cv2
import numpy as np
from experiment import motion
from utils import image as img_utils


if __name__ == '__main__':

    folder_path = "dataset/data_151_221/data (151)"

    imgs = img_utils.read_from_folder(folder_path)

    length = 75
    motion_list = []

    # XXX display input video
    # motion_list = imgs

    # XXX display motion blur
    # for i in range(len(imgs) - length):
    #     motion_list.append(motion.motion_blur(imgs[i:i + length]))

    # XXX display MEI
    for i in range(len(imgs) - length):
        motion_list.append(motion.motion_energy_image(imgs[i:i + length]))

    # XXX display MHI
    # for i in range(len(imgs) - length):
    #     motion_list.append(motion.motion_history_image(imgs[i:i + length]))

    # XXX display optical flow
    # motion_list = motion.optical_flow(imgs)

    img_utils.play_img_seq(motion_list, 30)
