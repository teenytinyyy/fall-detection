from typing import List
import cv2
import numpy as np


MAX_INTENSITY = 255


def motion_blur(img_arr: List[np.ndarray], coefficients: List[float] = []) -> np.ndarray:
    if not coefficients:
        coefficients = [1 for _ in range(len(img_arr))]

    for i in range(len(img_arr)):
        img_arr[i] = img_arr[i].astype(np.float32) * coefficients[i]

    motion_img = np.sum(img_arr, axis=0) / len(img_arr)

    return motion_img.astype(np.uint8)


def diff(img_a: np.ndarray, img_b: np.ndarray, thres: float = 0.2, max_intensity: int = MAX_INTENSITY) -> np.ndarray:
    diff = abs(img_a.astype(np.float32) - img_b.astype(np.float32))  # type: ignore

    diff[diff < max_intensity * thres] = 0

    return diff.astype(np.uint8)


def get_diff_sequence(img_arr: List[np.ndarray], step: int = 1, interval: int = 5) -> List[np.ndarray]:

    diff_list = []

    for i in range(0, len(img_arr) - interval, step):
        diff_list.append(diff(img_arr[i], img_arr[i + interval]))

    return diff_list


def binary_cumulation(binary_img_arr: List[np.ndarray], weights: List[float] = [], max_intensity: int = MAX_INTENSITY) -> np.ndarray:

    if len(weights) == 0:
        weights = [1 for _ in range(len(binary_img_arr))]

    cumulation_img = np.zeros_like(binary_img_arr[0])

    for img, w in zip(binary_img_arr, weights):
        img[img > 0] = max_intensity
        cumulation_img = np.maximum(cumulation_img, img * w)

    return cumulation_img


def motion_energy_image(img_arr: List[np.ndarray], step: int = 5, interval: int = 5, max_intensity: int = MAX_INTENSITY) -> np.ndarray:
    diff_list = get_diff_sequence(img_arr, step, interval)

    mei = binary_cumulation(diff_list).astype(np.uint8)

    mei[mei > 0] = max_intensity

    return mei


def motion_history_image(img_arr: List[np.ndarray], step: int = 5, interval: int = 5, background_decay: int = 20, max_intensity: int = MAX_INTENSITY) -> np.ndarray:
    diff_list = get_diff_sequence(img_arr, step, interval)

    decay_intensities = history_weights(len(diff_list)) * max_intensity

    cumulation_img = np.zeros_like(diff_list[0], dtype=np.float32)

    for img, intensity in zip(diff_list, decay_intensities):
        cumulation_img[img > 0] = intensity
        cumulation_img[img == 0] -= background_decay
        cumulation_img[cumulation_img < 0] = 0

    return cumulation_img.astype(np.uint8)


def history_weights(length: int):
    return np.arange(0, 1, 1 / length, dtype=np.float32)
