import cv2
import csv
import numpy as np

from utils import files as file_utils


def img_label(input_path: str, target_start: int, target_end: int, motion_num: int):
    img_all = []
    img_target = []
    img_paths = file_utils.get_files(input_path)

    for img_path in img_paths:
        # print(target_start, target_end)
        img = cv2.imread(img_path)
        img_idx, _ = file_utils.get_extension(file_utils.get_filename(img_path))
        h, w = img.shape[:2]
        if np.sum(img) >= h * w * 255 / 100 and target_start <= int(img_idx) <= target_end - motion_num:
            img_all.append(img_path)
            img_target.append(1)
        elif np.sum(img) >= h * w * 255 / 100:
            img_all.append(img_path)
            img_target.append(0)

    return img_all, img_target


if __name__ == "__main__":

    INPUT_PATH = "../motion/train_data ({})"
    OUTPUT_PATH = "../excel/motion_label.csv"
    target_start = []
    target_end = []
    target = "target.csv"

    with open(target, "r") as r_file:
        rows = csv.reader(r_file)
        for row in rows:
            target = [float(val) for val in row]
            target_start.append(target[1])
            target_end.append(target[2])

    img_list = []
    label_list = []
    writer_list = []
    for i in range(1, 25):
        images, labels = img_label(INPUT_PATH.format(i), target_start[i-1], target_end[i-1], 4)
        img_list += images
        label_list += labels
    for j in range(len(img_list)):
        writer_list.append([img_list[j], label_list[j]])

    with open(OUTPUT_PATH, "w") as w_file:
        writer = csv.writer(w_file)
        writer.writerows(writer_list)
