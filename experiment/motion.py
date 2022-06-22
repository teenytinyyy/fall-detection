from typing import List
import numpy as np


def motion_blur(img_arr: List[np.ndarray], coefficients: List[float] = []) -> np.ndarray:
    if not coefficients:
        coefficients = [1 for _ in range(len(img_arr))]

    for i in range(len(img_arr)):
        img_arr[i] = img_arr[i].astype(np.float32) * coefficients[i]

    motion_img = np.sum(img_arr, axis=0) / len(img_arr)

    return motion_img.astype(np.uint8)


def diff(img_a: np.ndarray, img_b: np.ndarray, thres: float = 0.2, max_intensity: int = 255) -> np.ndarray:
    diff = abs(img_a.astype(np.float32) - img_b.astype(np.float32))  # type: ignore

    diff[diff < max_intensity * thres] = 0

    return diff.astype(np.uint8)


def binary_cumulation(binary_img_arr: List[np.ndarray]) -> np.ndarray:

    cumulation_img = np.zeros_like(binary_img_arr[0])

    for img in binary_img_arr:
        cumulation_img = np.logical_or(cumulation_img, img)

    return cumulation_img


def motion_energy_image(img_arr: np.ndarray) -> np.ndarray:
    diff_list = []
    for i in range(1, len(img_arr)):
        diff_list.append(diff(img_arr[i], img_arr[i - 1]))

    mei = binary_cumulation(diff_list).astype(np.uint8)

    mei[mei > 0] = 255

    return mei
