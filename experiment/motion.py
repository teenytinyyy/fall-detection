from typing import List
import numpy as np


def motion_blur(img_arr: List[np.ndarray], coefficients: List[float] = []) -> np.ndarray:
    if not coefficients:
        coefficients = [1 for _ in range(len(img_arr))]

    for i in range(len(img_arr)):
        img_arr[i] = img_arr[i].astype(np.float32) * coefficients[i]

    motion_img = np.sum(img_arr, axis=0) / len(img_arr)

    return motion_img.astype(np.uint8)


def diff(img_a: np.ndarray, img_b: np.ndarray, thres: int = 5) -> np.ndarray:
    diff = abs(img_a.astype(np.float32) - img_b.astype(np.float32))  # type: ignore

    diff[diff < thres] = 0

    return diff.astype(np.uint8)
