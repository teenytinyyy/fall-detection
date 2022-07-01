import numpy as np
import cv2
import json

from utils import label as label_utils
from utils import image as img_utils
from utils import files as file_utils


if __name__ == '__main__':
    data_dir = "./dataset/data/json/json_data (1)"
    
    json_files = file_utils.get_sorted_files(data_dir, ext=[".json"])

    json_data_list = []

    for file_path in json_files:
        with open(file_path, "r") as reader:
            data = json.loads(reader.read())
            points = [[point[1], point[0]] for point in data['shapes'][0]['points']]
            json_data_list.append(points)

    img_list = img_utils.read_from_folder(data_dir)

    for idx, frame in enumerate(img_list):

        rot_angle, area, width, height, center_point, corner_points = label_utils.min_bounding_rect(json_data_list[idx])

        # print(rot_angle, area, width, height, center_point, corner_points)

        angle, width, height, center = label_utils.ellipse_bbox(json_data_list[idx])

        # print(angle, width, height, center)

        img_utils.draw_dot(frame, center[1], center[0], radius=7, color=(255, 255, 0))

        for point in corner_points:
            img_utils.draw_dot(frame, point[1], point[0], radius=5, color=(255, 255, 0))

        for point in json_data_list[idx]:
            img_utils.draw_dot(frame, point[1], point[0], radius=2, color=(255, 255, 0))

        cv2.ellipse(frame, (center[1], center[0]), (int(height), int(width)), -angle, 0, 360, (255, 255, 0), 1)
        
        cv2.imwrite("./dataset/data/json/json_data (1)/{}.jpg".format(idx), frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
