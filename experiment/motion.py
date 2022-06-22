from typing import List
import numpy as np
import cv2


MAX_INTENSITY = 255


def motion_blur(img_arr: List[np.ndarray], step: int = 5, interval: int = 5, max_intensity: int = MAX_INTENSITY) -> np.ndarray:

    diff_list = get_diff_sequence(img_arr, step, interval)

    for i in range(len(diff_list)):
        diff_list[i][diff_list[i] > 0] = max_intensity

    motion_img = np.sum(diff_list, axis=0) / len(diff_list)

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


def optical_flow(img_arr: List[np.ndarray], max_intensity: int = MAX_INTENSITY) -> List[np.ndarray]:

    first_frame = img_arr[0]
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(first_frame)

    # Sets image saturation to maximum
    mask[..., 1] = max_intensity

    representation_list = []

    for frame in img_arr:

        # Converts each frame to grayscale - we previously
        # only converted the first frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # gray = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)

        # Calculates dense optical flow by Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow
        # direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow
        # magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore

        # Converts HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        representation_list.append(rgb)

        # Updates previous frame
        prev_gray = gray

    return representation_list


def get_dynamic_image(frames: np.ndarray, normalized=True):
    """ Takes a list of frames and returns either a raw or normalized dynamic image."""
    num_channels = frames[0].shape[2]
    channel_frames = _get_channel_frames(frames, num_channels)
    channel_dynamic_images = [_compute_dynamic_image(channel) for channel in channel_frames]

    dynamic_image = cv2.merge(tuple(channel_dynamic_images))
    if normalized:
        dynamic_image = cv2.normalize(dynamic_image, None, 0, 255)
        dynamic_image = dynamic_image.astype('uint8')

    return dynamic_image


def _get_channel_frames(iter_frames, num_channels):
    """ Takes a list of frames and returns a list of frame lists split by channel. """
    frames = [[] for _ in range(num_channels)]

    for frame in iter_frames:
        for channel_frames, channel in zip(frames, cv2.split(frame)):
            channel_frames.append(channel.reshape((*channel.shape[0:2], 1)))

    for i in range(len(frames)):
        frames[i] = np.array(frames[i])  # type: ignore

    return frames


def _compute_dynamic_image(frames):
    """ Adapted from https://github.com/hbilen/dynamic-image-nets """
    num_frames, h, w, depth = frames.shape

    # Compute the coefficients for the frames.
    coefficients = np.zeros(num_frames)
    for n in range(num_frames):
        cumulative_indices = np.array(range(n, num_frames)) + 1
        coefficients[n] = np.sum(((2 * cumulative_indices) - num_frames) / cumulative_indices)

    # Multiply by the frames by the coefficients and sum the result.
    x1 = np.expand_dims(frames, axis=0)
    x2 = np.reshape(coefficients, (num_frames, 1, 1, 1))
    result = x1 * x2
    return np.sum(result[0], axis=0).squeeze()
