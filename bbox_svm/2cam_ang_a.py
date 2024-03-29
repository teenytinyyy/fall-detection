import csv
import math
from tabnanny import check
import numpy as np
from typing import Dict, List
import random
from sklearn.svm import SVC
from sklearn.externals import joblib


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


def train_svm_classify(data: np.ndarray, target: np.ndarray, check_list: np.ndarray, rev_idx: Dict, train_ratio: float = 0.7):
    randomized_indices = random_indices(0, len(data))
    x_train = data[randomized_indices[0: int(train_ratio * len(data))]]
    y_train = target[randomized_indices[0: int(train_ratio * len(data))]]

    x_test = data[randomized_indices[int(train_ratio * len(data)): len(data)]]
    y_test = target[randomized_indices[int(train_ratio * len(data)): len(data)]]
    check_points = check_list[randomized_indices[int(train_ratio * len(data)): len(data)]]

    #clf = SVC()
    #clf.fit(x_train, y_train)
    #joblib.dump(clf,'svm_model_avg_list.pkl')
    clf = joblib.load('svm_model.pkl')
    y_predict = clf.predict(x_test)

    for i in range(len(y_predict)):
        if y_predict[i] == 1 and check_points[i][0] <= 0 and check_points[i][1] <= 1.5:
            y_predict[i] = 1
        else:
            y_predict[i] = 0

    diff_indices = np.where(y_test ^ y_predict > 0)[0]

    diff_dict = {}
    for i in diff_indices:
        restored_idx = randomized_indices[int(train_ratio * len(data)): len(data)][i]
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
    threshold = 0.52
    avg_range = 13
    angle_list = []
    ar = []
    ar_data = []
    box_top_center_x = []
    box_top_center_y = []
    box_bottom_right_x = []
    box_bottom_right_y = []
    check_points = []
    check_list = []
    fall_target = []
    fall_data = []
    labels = []
    target_start = []
    target_end = []
    unused_num = []
    idx = 0
    count = 0
    count_2 = 0
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
        file_path = "../dataset/data/excel/2cam/test_data (" + str(num) + ").csv"
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
                box_top_center_x.append(x_data[1])
                box_top_center_y.append(x_data[2])
                box_bottom_right_x.append(x_data[3])
                box_bottom_right_y.append(x_data[4])

            for i in range(len(box_top_center_x)):
                ang = math.atan2(
                    abs(box_bottom_right_y[i] - box_top_center_y[i]),
                    abs(box_bottom_right_x[i] - box_top_center_x[i]),
                )
                angle_list.append(ang)

            radius = 30
            frame_idx = 30

            # y_top_v = get_diff_list(y_top_dis)
            ang_v_list = get_diff_list(angle_list)
            ang_a_list = get_diff_list(ang_v_list)
            avg_list = get_abs_avg_list(ang_a_list, avg_range)


        while True:

            if len(ar) - 4 <= frame_idx or len(avg_list) - radius <= frame_idx:
                break

            if 2 > abs(ar[frame_idx] - ar[frame_idx + 4]) >= threshold:
            # if avg_list[frame_idx] >= threshold * 0.1:

                if (len(avg_list[frame_idx - radius: frame_idx + radius]) == radius * 2 and len(ar[frame_idx - radius: frame_idx + radius]) == radius * 2):
                    ar_data.append(
                        max(avg_list[frame_idx - radius:frame_idx + radius]))
                    ar_data.append(
                        min(avg_list[frame_idx - radius:frame_idx + radius]))
                    ar_data.append(
                        max(ar[frame_idx - radius:frame_idx + radius]))
                    ar_data.append(
                        min(ar[frame_idx - radius:frame_idx + radius]))
                    #print(ar_data)
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

                    if (frame_idx - radius <= target_start[num - 1] + target_range <= frame_idx + radius):
                        # if check_points[-1] > 1:
                        #     fall_target.append(2)                
                        #     count_2 += 1

                        # else:                                
                        fall_target.append(1)
                        print("video:", num)
                        print("picture:", frame_idx)
                        print("label:", 1)
                        print("y", check_points)
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
                ar_data = []
                check_points = []

                continue


            frame_idx += 1

        ar = []
        angle_list = []
        ang_v_list = []
        ang_a_list = []
        avg_list = []
        box_top_center_x = []
        box_top_center_y = []
        box_bottom_right_x = []
        box_bottom_right_y = []

    print("fall_target", len(fall_target))
    print("thres", count)
    print("fall", sum(fall_target))
    print("count_2", count_2)

    X_1 = np.array(fall_data)
    Y = np.array(fall_target)
    check_list = np.array(check_list)


    train_svm_classify(X_1, Y, check_list, rev_idx, 0.7)
    


'''
    random_idx = random_indices(0, len(X_1))
    x_1_train = X_1[random_idx[0:int(0.7 * len(X_1))]]
    y_1_train = Y_1[random_idx[0:int(0.7 * len(X_1))]]
    x_1_test = X_1[random_idx[int(0.7 * len(X_1)):len(X_1)]]
    y_1_test = Y_1[random_idx[int(0.7 * len(X_1)):len(X_1)]]

    x_2_train = X_2[random_idx[0:int(0.7 * len(X_2))]]
    x_2_test = X_2[random_idx[int(0.7 * len(X_2)):len(X_2)]]

    clf = SVC()
    clf.fit(x_1_train, y_1_train)
    # joblib.dump(clf, '../states/2cam.pkl')
    clf_2 = SVC()
    clf_2.fit(x_2_train, y_1_train)

    y_1_predict = clf.predict(x_1_test)
    y_2_predict = clf_2.predict(x_2_test)
    # for i in range(len(y_1_predict)):
    #     if y_1_predict[i] == 1 and y_2_predict[i] == 0:
    #         y_1_predict[i] = 0
    # elif y_1_predict[i] == 0 and y_2_predict[i] == 1:
    #     y_1_predict[i] = 1
    # print(i)
    diff_idx = np.where(y_1_test ^ y_1_predict > 0)[0]

    print("diff indices:")
    print(diff_idx)

    diff_dict = {}
    for i in diff_idx:
        restored_idx = random_idx[int(0.7 * len(X_1)):len(X_1)][i]
        diff_dict[restored_idx] = [restored_idx, y_1_predict[i], y_1_test[i]]
        print(rev_idx[restored_idx], diff_dict[restored_idx])


    print("Diff count:{}".format(len(diff_dict.keys())))

    idx_list = np.array(
        random_idx[int(0.7 * len(X_1)):len(X_1)])[diff_idx].astype(np.uint8)

    # print("source info")
    # for i in idx_list:
    #     print(i, rev_idx[i])

    # print("XOR", y_test^y_predict)
    # plt.scatter(x_test[:,0], x_test[:,1], c=y_predict)
    # plt.savefig("t.png")

    # for k, v in rev_idx.items():
    #     print("rev", k, v)

    confusion_mat = np.zeros((2, 2))
    idx_pairs = np.array([y_1_predict.tolist(), y_1_test.tolist()]).T.tolist()
    a = 0
    for idx_pair in idx_pairs:
        # print("gt", Y[random_idx[int(0.7 * len(X)):len(X)][a]], y_test[a])
        confusion_mat[tuple(idx_pair)] += 1
        a += 1

    print("confusion matrix")
    print(confusion_mat)
    result = y_1_test ^ y_1_predict

    print(sum(Y_1))
    print("Accuracy: {:2f}% {} / {} / {}".format((1 - (result.sum() /
          len(result))) * 100, sum(y_1_test), result.sum(), len(y_1_test)))
'''
