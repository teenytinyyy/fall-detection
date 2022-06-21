
from typing import List
import numpy as np
import cv2


def get_dynamic_image(frames, normalized: bool = False):
    """ Takes a list of frames and returns either a raw or normalized dynamic image."""
    num_channels = frames[0].shape[2]
    channel_frames = _get_channel_frames(frames, num_channels)
    channel_dynamic_images = [_compute_dynamic_image(channel) for channel in channel_frames]

    dynamic_image = cv2.merge(tuple(channel_dynamic_images))

    print(dynamic_image.dtype)

    if normalized:
        dynamic_image = cv2.normalize(dynamic_image, None, 0, 255)  # type: ignore
        dynamic_image = dynamic_image.astype('uint8')

    return dynamic_image.astype(np.uint8)


def _get_channel_frames(iter_frames, num_channels):
    """ Takes a list of frames and returns a list of frame lists split by channel. """
    frames = [[] for _ in range(num_channels)]

    for frame in iter_frames:
        for channel_frames, channel in zip(frames, cv2.split(frame)):
            channel_frames.append(channel.reshape((*channel.shape[0:2], 1)))

    for i in range(len(frames)):
        frames[i] = np.array(frames[i])

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
