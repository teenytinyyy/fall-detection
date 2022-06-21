from typing import List
from natsort import natsorted
import cv2
import glob
import numpy as np

from utils import files as file_utils


def batch_read_img(file_name: str):
    img_list = [cv2.imread(str(file)) for file in natsorted(glob.glob(file_name + "/*.jpg"))]
    return img_list


def read_from_folder(folder_path: str) -> List[np.ndarray]:
    img_path_list = file_utils.get_files(folder_path)
    img_list = []

    def filename_key(x: str):
        filename, _ = file_utils.get_extension(file_utils.get_filename(x))
        filename = filename.zfill(4)
        return filename

    img_path_list.sort(key=filename_key)

    for img_path in img_path_list:
        img_list.append(cv2.imread(img_path))

    return img_list


def diff(img_a: np.ndarray, img_b: np.ndarray, tolerance: int = 5) -> np.ndarray:
    diff = np.abs(img_a.astype(np.float32) - img_b.astype(np.float32))  # type: ignore
    diff[diff < tolerance] = 0
    return diff.astype(np.uint8)


def motion_blur(img_arr: np.ndarray, coefficients: List[float] = []) -> np.ndarray:
    if not coefficients:
        coefficients = [1 for _ in range(len(img_arr))]

    for i in range(len(img_arr)):
        img_arr[i] = img_arr[i].astype(np.uint8) * coefficients[i]

    motion_img = np.sum(img_arr, axis=0) / len(img_arr)

    return motion_img.astype(np.uint8)


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
