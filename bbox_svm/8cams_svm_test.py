import csv
import math
from tabnanny import check
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


def test_svm_classify(data: np.ndarray, target: np.ndarray, check_list: np.ndarray, rev_idx: Dict, train_ratio: float = 0.7):
    randomized_indices = random_indices(0, len(data))
    x_train = data[randomized_indices[0: int(train_ratio * len(data))]]
    y_train = target[randomized_indices[0: int(train_ratio * len(data))]]

    x_test = data[randomized_indices[int(train_ratio * len(data)): len(data)]]
    y_test = target[randomized_indices[int(train_ratio * len(data)): len(data)]]
    check_points = check_list[randomized_indices[int(train_ratio * len(data)): len(data)]]

    clf = SVC()
    clf.fit(x_train, y_train)

    y_predict = clf.predict(x_test)

    # for i in range(len(y_predict)):
    #     if y_predict[i] == 1 and check_points[i][0] <= 0 and check_points[i][1] <= 1.5:
    #         y_predict[i] = 1
    #     else:
    #         y_predict[i] = 0

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

    target = "../dataset/data/excel/8cams_labels.csv"
    delay = "../dataset/data/excel/8cams_delay.csv"
    video_start = 4
    video_end = 24
    target_range = 13
    threshold = 0.52
    avg_range = 13

    # count_2 = 0


    delay_nums = []
    with open(delay, "r") as r_file:
        rows = csv.reader(r_file)
        for row in rows:
            delay = [int(val) for val in row]
            delay_nums.append(row)

    check_list = []
    fall_target = []
    fall_data = []
    rev_idx = {}
    count = 0


    for num in range(video_start, video_end + 1):
        for num1 in range(1, 9):
            file_path = "../dataset/data/excel/mask_rcnn_1286/" + str(num) + "_" + str(num1) + ".csv"
            delay_num = delay_nums[num - 1][num1 - 1]
            print(num, num1)
            with open(file_path, "r") as r_file:
                rows = csv.reader(r_file)
                ar = []
                angle_list = []
                box_top_center_y = []
                ang_v_list = []
                ang_a_list = []
                avg_list = []
                frame_num = []

                for row in rows:
                    x_data = [float(val) for val in row]   #x1,y1,x2,y2

                    if (x_data[8] - int(delay_num)) <= 0:
                        continue
                    else:                        
                        frame_num.append(x_data[8] - float(delay_num))       
                        ar.append((x_data[3] - x_data[1]) / (x_data[2] - x_data[0]))
                        box_top_center_y.append((x_data[3] + x_data[1]) / 2)                
                        ang = math.atan2(
                            abs((x_data[3] - x_data[1]) / 2),
                            abs(x_data[2] - x_data[0]),
                        )
                        angle_list.append(ang)
 
            target_starts = []
            target_ends = []
            target_labels = []
            with open(target, "r") as r_file:
                rows = csv.reader(r_file)
                for row in rows:
                    target_list = [int(val) for val in row]
                    if target_list[0] == num:
                        target_starts.append(target_list[1])
                        target_ends.append(target_list[2])
                        target_labels.append(target_list[3])

            target_idx = 0
            radius = 30
            frame_idx = 30
            ang_v_list = get_diff_list(angle_list)
            ang_a_list = get_diff_list(ang_v_list)
            avg_list = get_abs_avg_list(ang_a_list, avg_range)


            while True:
                
                target_end = target_ends[target_idx]
                if frame_idx > target_end:
                    target_idx += 1
                if target_idx == len(target_starts):
                    break                
                target_start = target_starts[target_idx]
                target_end = target_ends[target_idx]
                target_label = target_labels[target_idx]

                if len(ar) - 4 <= frame_idx or len(avg_list) - radius <= frame_idx:
                    break

                if 2 > abs(ar[frame_idx] - ar[frame_idx + 4]) >= threshold:
                    ar_data = []
                    check_points = []

                    if (len(avg_list[frame_idx - radius: frame_idx + radius]) == radius * 2 and len(ar[frame_idx - radius: frame_idx + radius]) == radius * 2) and frame_idx <= target_end:
                        print(frame_idx, target_start, target_end, target_label)
                        #ar_data.append(
                        #    max(avg_list[frame_idx - radius:frame_idx + radius]))
                        # ar_data.append(
                        #     min(avg_list[frame_idx - radius:frame_idx + radius]))
                        ar_data.append(
                            max(ar[frame_idx - radius:frame_idx + radius]))
                        ar_data.append(
                            min(ar[frame_idx - radius:frame_idx + radius]))
                        ar_data.append(
                            max(angle_list[frame_idx - radius:frame_idx + radius]))
                        # ar_data.append(
                        #    min(angle_list[frame_idx - radius:frame_idx + radius]))

                        check_points.append(box_top_center_y[frame_idx - radius] - box_top_center_y[frame_idx + radius])
                        check_points.append(ar[frame_idx + radius])

                        fall_data.append(ar_data)
                        check_list.append(check_points)


                        # if check_points[-1] > 1:
                        #     fall_target.append(0)
                        #     frame_idx += radius
                        #     print("video:", num)
                        #     print("picture:", frame_idx)
                        #     print("label:", 0)

                        if target_label == 2:
                            # if check_points[-1] > 1:
                            #     fall_target.append(2)                
                            #     count_2 += 1

                            # else:                                
                            fall_target.append(1)
                            print("video:", num)
                            print("picture:", frame_idx)
                            print("label:", 1)
                            print("angle", max(angle_list[frame_idx - radius:frame_idx + radius]))
                            print(ar[frame_idx + radius])

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
                    print('a')

                    continue


                frame_idx += 1



    print("fall_target", len(fall_target))
    print("thres", count)
    print("fall", sum(fall_target))
    # print("count_2", count_2)

    X_1 = np.array(fall_data)
    Y = np.array(fall_target)
    check_list = np.array(check_list)


    test_svm_classify(X_1, Y, check_list, rev_idx, 0.7)