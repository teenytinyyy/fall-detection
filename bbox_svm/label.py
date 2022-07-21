from array import array
from pickle import TUPLE
from typing import List
from imantics import Mask
import cv2
import csv
import numpy as np
import math
import sys

import files as file_utils

MAX_INT = 1e10


def img_label(input_path: str, target_start: int, target_end: int, motion_num: int):
    img_all = []
    img_target = []
    img_paths = file_utils.get_files(input_path)

    for img_path in img_paths:
        img = cv2.imread(img_path)
        img_idx, _ = file_utils.get_extension(file_utils.get_filename(img_path))
        h, w = img.shape[:2]
        if np.sum(img) >= h * w * 255 / 100 and target_start <= int(img_idx) <= target_end - motion_num:
            img_all.append(img_path)
            img_target.append(1)
        elif np.sum(img) >= h * w * 255 / 100:
            img_all.append(img_path)
            img_target.append(0)

    return img_all, img_target


def clockwise_angle_and_distance(point, origin=[0, 0], ref_vec=[1, 0]):

    vector = [point[0] - origin[0], point[1] - origin[1]]

    len_vector = math.hypot(vector[0], vector[1])

    if len_vector == 0:
        return -math.pi, 0

    normalized = [vector[0] / len_vector, vector[1] / len_vector]
    dot_prod = normalized[0] * ref_vec[0] + normalized[1] * ref_vec[1]
    diff_prod = ref_vec[1] * normalized[0] - ref_vec[0] * normalized[1]
    angle = math.atan2(diff_prod, dot_prod)

    if angle < 0:
        return 2 * math.pi + angle, len_vector

    return angle, len_vector


def min_bounding_rect(points: np.ndarray):

    origin = [np.average(points, axis=0)[0], np.average(points, axis=0)[1]]

    def clockwise_key(pos):
        return clockwise_angle_and_distance(pos, origin)

    points = np.array(sorted(list(points), key=clockwise_key))

    # Compute edges (x2-x1,y2-y1)
    edges = np.zeros((len(points) - 1, 2))  # empty 2 column array
    for i in range(len(edges)):
        edge_x = points[i + 1, 0] - points[i, 0]
        edge_y = points[i + 1, 1] - points[i, 1]
        edges[i] = [edge_x, edge_y]

    # Calculate edge angles   atan2(y/x)
    edge_angles = np.zeros((len(edges)))  # empty 1 column array
    for i in range(len(edge_angles)):
        edge_angles[i] = math.atan2(edges[i, 1], edges[i, 0])

    # Check for angles in 1st quadrant
    for i in range(len(edge_angles)):
        edge_angles[i] = abs(edge_angles[i] % (math.pi / 2))  # want strictly positive answers

    # Remove duplicate angles
    edge_angles = np.unique(edge_angles)

    # Test each angle to find bounding box with smallest area
    min_bbox = (0, MAX_INT, 0, 0, 0, 0, 0, 0)  # rot_angle, area, width, height, min_x, max_x, min_y, max_y
    for i in range(len(edge_angles)):

        # Create rotation matrix to shift points to baseline
        # R = [ cos(theta)      , cos(theta-PI/2)
        #       cos(theta+PI/2) , cos(theta)     ]
        R = np.array([[math.cos(edge_angles[i]), math.cos(edge_angles[i] - (math.pi / 2))],
                      [math.cos(edge_angles[i] + (math.pi / 2)), math.cos(edge_angles[i])]])

        # Apply this rotation to convex hull points
        rot_points = np.dot(R, np.transpose(points))  # 2x2 * 2xn

        # Find min/max x,y points
        min_x = np.nanmin(rot_points[0], axis=0)
        max_x = np.nanmax(rot_points[0], axis=0)
        min_y = np.nanmin(rot_points[1], axis=0)
        max_y = np.nanmax(rot_points[1], axis=0)

        # Calculate height/width/area of this bounding rectangle
        width = max_x - min_x
        height = max_y - min_y
        area = width * height

        # Store the smallest rect found first (a simple convex hull might have 2 answers with same area)
        if (area < min_bbox[1]):
            min_bbox = (edge_angles[i], area, width, height, min_x, max_x, min_y, max_y)
        # Bypass, return the last found rect

    # Re-create rotation matrix for smallest rect
    angle = min_bbox[0]
    R = np.array([[math.cos(angle), math.cos(angle - (math.pi / 2))], [math.cos(angle + (math.pi / 2)), math.cos(angle)]])

    # min/max x,y points are against baseline
    min_x = min_bbox[4]
    max_x = min_bbox[5]
    min_y = min_bbox[6]
    max_y = min_bbox[7]

    # Calculate center point and project onto rotated frame
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_point = np.dot([center_x, center_y], R)

    # Calculate corner points and project onto rotated frame
    corner_points = np.zeros((4, 2))  # empty 2 column array
    corner_points[0] = np.dot([max_x, min_y], R)
    corner_points[1] = np.dot([min_x, min_y], R)
    corner_points[2] = np.dot([min_x, max_y], R)
    corner_points[3] = np.dot([max_x, max_y], R)

    angle = (angle / np.pi) * 180

    return (angle, min_bbox[1], min_bbox[2], min_bbox[3], center_point, corner_points)  # rot_angle, area, width, height, center_point, corner_points


def ellipse_bbox(points: np.ndarray):

    angle, _, width, height, center, _ = min_bounding_rect(points)

    center = [int(pos) for pos in center]

    return (angle, width / 2, height / 2, center)


def mask2ellipse(mask_array: list):
    polygons = Mask(mask_array).polygons()

    polygons_points = np.array(polygons.points)

    max_area = 0
    max_idx = 0
    for i in range(len(polygons_points)):
        area = cv2.contourArea(polygons_points[i])
        if area > max_area:
            max_area = area
            max_idx = i

    points = np.array([[point[1], point[0]] for point in polygons_points[max_idx]])


    # for debugging
    # print(polygons_points[max_idx])

    angle, width, height, center = ellipse_bbox(points)

    return angle, width, height, center, polygons_points[max_idx]
