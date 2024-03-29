from typing import List, Optional
from natsort import natsorted
import cv2
import glob
import numpy as np

import files as file_utils


def batch_read_img(file_name: str):
    img_list = [cv2.imread(str(file))
                for file in natsorted(glob.glob(file_name + "/*.jpg"))]
    return img_list


def read_from_folder(folder_path: str) -> List[np.ndarray]:
    img_path_list = file_utils.get_files(
        folder_path, [".jpg", ".png", ".jpeg"])

    img_list = []

    def filename_key(x: str):
        filename, _ = file_utils.get_extension(file_utils.get_filename(x))
        filename = filename.zfill(4)
        return filename

    img_path_list.sort(key=filename_key)

    for img_path in img_path_list:
        img_list.append(cv2.imread(img_path))

    return img_list


def resize(img: np.ndarray, new_size: int):

    h, w, c = img.shape
    img_size = max(h, w)
    img_new = np.zeros((img_size, img_size, c)).astype(np.uint8)

    img_new[(w - h) // 2:w - (w - h) // 2, 0:w] = img
    img_new = cv2.resize(img_new, (new_size, new_size))
    return img_new


def play_img_seq(img_list: List[np.ndarray], frame_per_ms: int = 10) -> None:

    for idx, frame in enumerate(img_list):

        print("processing image {}/{}".format(idx, len(img_list)))

        cv2.imshow("image sequence", frame)

        if cv2.waitKey(frame_per_ms) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def write_img_seq(img_list: List[np.ndarray], dir: str) -> None:

    for idx, frame in enumerate(img_list):

        print("processing image {}/{}".format(idx, len(img_list)))

        cv2.imwrite("{}/img_seq_{}.jpg".format(dir, idx), frame)


def draw_dot(img, x, y, radius=2, color=(0, 0, 255)):
    cv2.circle(img, (int(x), int(y)), radius=radius, color=color, thickness=-1)


def add_text(img, text, x, y):
    cv2.putText(img, text=text, org=(int(x), int(y)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(0, 255, 255), lineType=cv2.LINE_AA)


def display_bbox(img, x, y, w, h, output_path: Optional[str] = None, ratio_mode: bool = True):

    if ratio_mode:
        img_height, img_width, _ = img.shape
        rect_start = (int((float(x) - float(w) / 2) * img_width),
                      int((float(y) - float(h) / 2) * img_height))
        rect_end = (int((float(x) + float(w) / 2) * img_width),
                    int((float(y) + float(h) / 2) * img_height))
    else:
        rect_start = (int(float(x) - float(w) / 2),
                      int(float(y) - float(h) / 2))
        rect_end = (int(float(x) + float(w) / 2), int(float(y) + float(h) / 2))

    cv2.rectangle(img, rect_start, rect_end, (0, 0, 255), 1)

    if output_path is None:
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        cv2.imwrite(output_path, img)


def write_video(img_list: List[np.ndarray], des: str, frame_rate: int = 25):
    h, w, _ = img_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(des, fourcc, frame_rate, (w, h))

    for frame in img_list:
        writer.write(frame)

    writer.release()


def crop_area(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, expansion_range: int = 0):

    y1 = y1 - expansion_range
    y2 = y2 + expansion_range
    x1 = x1 - expansion_range
    x2 = x2 + expansion_range
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > len(img[0]):
        x2 = len(img[0])
    if y2 > len(img):
        y2 = len(img)
    crop_img = img[y1:y2, x1:x2]


    return crop_img, x1, y1

def calculate_IOU(predicted_bound, ground_truth_bound):
    px1, py1, px2, py2 = predicted_bound
    gx1, gy1, gx2, gy2 = ground_truth_bound
    parea = (px2 - px1) * (py2 - py1)
    garea = (gx2 - gx1) * (gy2 - gy1)
    
    # 相交矩形座標
    x1 = max(px1, gx1)
    y1 = max(py1, gy1)
    x2 = min(px2, gx2)
    y2 = min(py2, gy2)

    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return 0

    area = w * h
    IoU = area / (parea + garea - area)

    return IoU
