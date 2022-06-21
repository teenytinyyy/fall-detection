from typing import List
import cv2
import numpy as np


def gen_mhi(frames: List[np.ndarray], thres: int = 32, duration: int = 50) -> np.ndarray:

    frame = frames[0]

    h, w = frame.shape[:2]
    prev_frame = frame.copy()

    motion_history = np.zeros((h, w), np.float32)

    timestamp = 0
    for frame in frames:

        frame_diff = cv2.absdiff(frame, prev_frame)

        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)

        _, fg_mask = cv2.threshold(gray_diff, thres, 1, cv2.THRESH_BINARY)
        timestamp += 1

        # update motion history
        cv2.motempl.updateMotionHistory(fg_mask, motion_history, timestamp, duration)

        # normalize motion history
        mh = np.uint8(np.clip((motion_history - (timestamp - duration)) / duration, 0, 1) * 255)
        cv2.imshow('motion-history', mh)
        cv2.imshow('raw', frame)

        prev_frame = frame.copy()
        if 0xFF & cv2.waitKey(5) == 27:
            break

    cv2.destroyAllWindows()
