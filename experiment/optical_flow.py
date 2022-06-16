import cv2 as cv
import cv2
import numpy as np
from utils import files as file_utils


def display_optical_flow(folder_path: str, frame_per_ms: int = 10):
    img_path_list = file_utils.get_files(folder_path)

    def filename_key(x: str):
        filename, _ = file_utils.get_extension(file_utils.get_filename(x))
        filename = filename.zfill(4)
        return filename

    img_path_list.sort(key=filename_key)

    first_frame = cv2.imread(img_path_list[0])
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    mask = np.zeros_like(first_frame)

    # Sets image saturation to maximum
    mask[..., 1] = 255

    for idx, img_path in enumerate(img_path_list):

        print("processing image {} {}/{}".format(img_path, idx, len(img_path_list)))

        frame = cv2.imread(img_path)

        # Opens a new window and displays the input
        # frame
        cv.imshow("input", frame)

        # Converts each frame to grayscale - we previously
        # only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow
        # direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow
        # magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

        # Opens a new window and displays the output frame
        cv.imshow("dense optical flow", rgb)

        # Updates previous frame
        prev_gray = gray

        # Frames are read by intervals of 1 millisecond. The
        # programs breaks out of the while loop when the
        # user presses the 'q' key
        if cv.waitKey(frame_per_ms) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
