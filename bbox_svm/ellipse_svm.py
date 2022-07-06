import csv
import math
import numpy as np
from typing import Dict, List
import random
from sklearn.svm import SVC


def random_indices(start: int, stop: int, step: int = 1) -> List[int]:
    indices = [idx for idx in range(start, stop, step)]

    random.shuffle(indices)

    return indices


def get_diff_list(angle_list: List[float], diff_range: int = 4) -> List[float]:

    diff_list = []
    for idx in range(len(angle_list) - diff_range):
        diff = angle_list[idx] - angle_list[idx + diff_range]
        diff_list.append(diff)

    return diff_list


def get_abs_avg_list(angle_list: List[float], avg_range: int = 13) -> List[float]:

    np_angle_list = np.array(angle_list)

    avg_list = []
    for idx in range(len(np_angle_list) - avg_range):
        avg = np.average(np.absolute(np_angle_list[idx: idx + avg_range]))
        avg_list.append(avg)

    return avg_list


def get_avg_list(angle_list: List[float], avg_range: int = 13) -> List[float]:

    np_angle_list = np.array(angle_list)

    avg_list = []
    for idx in range(len(np_angle_list) - avg_range):
        # avg = np.average(np.absolute(np_angle_list[idx:idx + avg_range]))
        avg = np.average(np_angle_list[idx: idx + avg_range])
        avg_list.append(avg)

    return avg_list


def test_svm_classify(data: np.ndarray, target: np.ndarray, rev_idx: Dict, train_ratio: float = 0.7):
    randomized_indices = random_indices(0, len(data))
    x_train = data[randomized_indices[0: int(train_ratio * len(data))]]
    y_train = target[randomized_indices[0: int(train_ratio * len(data))]]

    x_test = data[randomized_indices[int(train_ratio * len(data)): len(data)]]
    y_test = target[randomized_indices[int(
        train_ratio * len(data)): len(data)]]

    clf = SVC()
    clf.fit(x_train, y_train)

    y_predict = clf.predict(x_test)
    diff_indices = np.where(y_test ^ y_predict > 0)[0]

    diff_dict = {}
    for i in diff_indices:
        restored_idx = randomized_indices[int(0.7 * len(data)): len(data)][i]
        restored_diff_item = [restored_idx, y_predict[i], y_test[i]]
        diff_dict[restored_idx] = restored_diff_item

        print("Diff at index {}: prediction: {} <-> ground truth: {}".format(
            rev_idx[restored_idx], restored_diff_item[1], restored_diff_item[2]))

    # builds confusion matrix
    confusion_mat = np.zeros((2, 2))
    idx_pairs = np.array([y_predict.tolist(), y_test.tolist()]).T.tolist()
    for idx_pair in idx_pairs:
        confusion_mat[tuple(idx_pair)] += 1

    print("confusion matrix")
    print(confusion_mat)


if __name__ == "__main__":

    target = "../dataset/data/excel/2cam/svm_2cam_target.csv"
    unused = "../dataset/data/excel/2cam/2cam.csv"
    video_start = 1
    video_end = 221
    target_range = 13
    threshold = 0.5
    avg_range = 13
    angle_list = []
    ar = []
    ar_data = []
    box_x = []
    box_y = []
    box_angle = []
    box_bottom_right_x = []
    box_bottom_right_y = []
    fall_target = []
    fall_data = []
    labels = []
    max_data = []
    min_data = []
    sec_data = []
    sec_fall_data = []
    N3_data = []
    N3_fall_data = []
    target_start = []
    target_end = []
    unused_num = []
    y_dis_list = []
    idx = 0
    count = 0
    count_unused = 0
    rev_idx = {}

    with open(target, "r") as r_file:
        rows = csv.reader(r_file)
        for row in rows:
            target = [float(val) for val in row]
            target_start.append(target[1])
            target_end.append(target[2])

    with open(unused, "r") as r_file:
        rows = csv.reader(r_file)
        for row in rows:
            unused_num.append(int(row[0]))

    for num in range(video_start, video_end + 1):
        file_path = "../dataset/data/excel/ellipse/data (" + str(num) + ").csv"
        # file = open(file_name, newline='')  with用完會幫你關，單用open 要自己關
        # rows = csv.reader(file)

        if num == unused_num[count_unused]:
            count_unused += 1
            continue

        with open(file_path, "r") as r_file:
            rows = csv.reader(r_file)
            for row in rows:
                x_data = [float(val) for val in row]
                ar.append(x_data[0])
                box_x.append(x_data[1])
                box_y.append(x_data[2])
                box_angle.append(x_data[3])

            ang_v_list = get_diff_list(box_angle)
            ang_a_list = get_diff_list(ang_v_list)
            avg_list = get_avg_list(ang_a_list, avg_range)

            radius = 30
            frame_idx = 30
            while True:

                if len(ar) - 4 <= frame_idx or len(avg_list) - radius <= frame_idx:
                    break

                if 2 > abs(ar[frame_idx] - ar[frame_idx + 4]) >= threshold:

                    if (len(avg_list[frame_idx - radius: frame_idx + radius]) == radius * 2 and len(ar[frame_idx - radius: frame_idx + radius]) == radius * 2):
                        ar_data.append(
                            max(avg_list[frame_idx - radius:frame_idx + radius]))
                        ar_data.append(
                            min(avg_list[frame_idx - radius:frame_idx + radius]))
                        ar_data.append(
                            max(ar[frame_idx - radius:frame_idx + radius]))
                        ar_data.append(
                            min(ar[frame_idx - radius:frame_idx + radius]))
                        ar_data.append(
                            max(box_angle[frame_idx - radius:frame_idx + radius]))
                        ar_data.append(
                            min(box_angle[frame_idx - radius:frame_idx + radius]))
                        print(max(box_angle[frame_idx - radius:frame_idx + radius]))
                        print(min(box_angle[frame_idx - radius:frame_idx + radius]))
                            
                        fall_data.append(ar_data)


                        if (frame_idx - radius <= target_start[num - 1] + target_range <= frame_idx + radius):
                            fall_target.append(1)
                            print("video:", num)
                            print("picture:", frame_idx)
                            print("label:", 1)

                            frame_idx += radius
                        else:
                            fall_target.append(0)
                            frame_idx += radius

                        rev_idx[len(fall_target) - 1] = (
                            num,
                            frame_idx - radius,
                            fall_target[-1],
                        )

                    count += 1
                    frame_idx += 1
                    ar_data = []

                    continue

                frame_idx += 1

            ar = []
            angle_list = []
            ang_v_list = []
            ang_a_list = []
            avg_list = []
            box_x = []
            box_y = []

            y_dis_list = []
            y_top_v = []
            y_top_a = []

    print(len(fall_target))
    print(sum(fall_target))
    print(count)

    X_1 = np.array(fall_data)
    Y = np.array(fall_target)

    test_svm_classify(X_1, Y, rev_idx, 0.7)
