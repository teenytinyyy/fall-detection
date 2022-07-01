import numpy as np
import cv2
import json

from utils import label as label_utils
from utils import image as img_utils
from utils import files as file_utils


if __name__ == '__main__':
    data_dir = "dataset/json_data (1)"
    json_files = file_utils.get_sorted_files(data_dir, ext=[".json"])
    img_files = file_utils.get_sorted_files(data_dir, ext=[".jpg"])
    labeled_files = []

    json_data_list = []
    img_list = []

    json_map = {}
    for json_path in json_files:
        json_idx, _ = file_utils.get_extension(file_utils.get_filename(json_path))
        json_map[json_idx] = True

    for img_path in img_files:
        img_idx, _ = file_utils.get_extension(file_utils.get_filename(img_path))
        if img_idx in json_map:
            labeled_files.append(img_path)

    for file_path in json_files:
        with open(file_path, "r") as reader:
            data = json.loads(reader.read())
            points = [[point[1], point[0]] for point in data['shapes'][0]['points']]
            json_data_list.append(points)

    for img_path in labeled_files:
        img = cv2.imread(img_path)
        img_list.append(img)

    for idx, frame in enumerate(img_list):

        rot_angle, area, width, height, center_point, corner_points = label_utils.min_bounding_rect(json_data_list[idx])

        # print(rot_angle, area, width, height, center_point, corner_points)

        angle, width, height, center = label_utils.ellipse_bbox(json_data_list[idx])

        # print(angle, width, height, center)

        img_utils.draw_dot(frame, center[1], center[0], radius=2, color=(255, 255, 0))

        for point in corner_points:
            img_utils.draw_dot(frame, point[1], point[0], radius=2, color=(255, 255, 0))

        for point in json_data_list[idx]:
            img_utils.draw_dot(frame, point[1], point[0], radius=2, color=(255, 255, 0))

        cv2.ellipse(frame, (center[1], center[0]), (int(height), int(width)), -angle, 0, 360, (255, 255, 0), 5)

        cv2.imshow("image sequence", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
