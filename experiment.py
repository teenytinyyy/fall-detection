import cv2
import numpy as np
from experiment import motion, optical_flow
from utils import image as img_utils


if __name__ == '__main__':
    folder_path = "dataset/data_151_221/data (151)"

    # optical_flow.display_optical_flow(folder_path)

    imgs = np.array(img_utils.read_from_folder(folder_path))

    step = 5
    diff_list = []
    for i in range(len(imgs) - step):
        diff_list.append(motion.diff(imgs[i], imgs[i + step]))

    length = 5
    motion_list = []
    for i in range(len(diff_list) - length):
        motion_list.append(motion.motion_blur(diff_list[i:i + length]))

    img_utils.play_img_seq(motion_list)
