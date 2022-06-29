import numpy as np

from utils import label as label_utils


if __name__ == '__main__':
    points = np.array([
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (3, 1)
    ])

    result = label_utils.ellipse_bbox(points)

    print(result)
