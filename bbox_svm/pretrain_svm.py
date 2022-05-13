import csv
import math
import numpy as np
from typing import List


def get_diff_list(angle_list: List[float], diff_range: int = 4) -> List[float]:

    diff_list = []
    for idx in range(len(angle_list) - diff_range):
        diff = angle_list[idx] - angle_list[idx + diff_range]
        diff_list.append(diff)

    return diff_list


def get_avg_list(angle_list: List[float], avg_range: int = 13) -> List[float]:

    np_angle_list = np.array(angle_list)

    avg_list = []
    for idx in range(len(np_angle_list) - avg_range):
        avg = np.average(np.absolute(np_angle_list[idx:idx + avg_range]))
        avg_list.append(avg)

    return avg_list


if __name__ == '__main__':

    target = "./excel/target.csv"
    video_start = 1
    video_end = 221
    target_range = 7
    threshold = 0.6
    avg_range = 13
    angle_list = []
    ar = []
    ar_data = []
    box_top_center_x = []
    box_top_center_y = []
    box_buttom_right_x = []
    box_buttom_right_y = []
    fall_target = []
    fall_data = []
    labels = []
    max_data = []
    min_data = []
    target_start = []
    target_end = []
    idx = 0
    count = 0

    with open(target, "r") as r_file:
        rows = csv.reader(r_file)
        for row in rows:
            target = [float(val) for val in row]
            target_start.append(target[1])
            target_end.append(target[2])

    for num in range(video_start, video_end+1):
        file_path = "./excel/Home_01/data (" + str(num) + ").csv"
        # file = open(file_name, newline='')    with用完會幫你關，單用open 要自己關
        # rows = csv.reader(file)

        with open(file_path, "r") as r_file:
            rows = csv.reader(r_file)
            for row in rows:
                x_data = [float(val) for val in row]
                ar.append(x_data[0])
                box_top_center_x.append(x_data[1])
                box_top_center_y.append(x_data[2])
                box_buttom_right_x.append(x_data[3])
                box_buttom_right_y.append(x_data[4])

            for i in range(len(box_top_center_x)):
                ang = math.atan2(abs(box_buttom_right_y[i] - box_top_center_y[i]), abs(
                    box_buttom_right_x[i] - box_top_center_x[i]))
                angle_list.append(ang)

            ang_v_list = get_diff_list(angle_list)
            ang_a_list = get_diff_list(ang_v_list)
            avg_list = get_avg_list(ang_a_list, avg_range)

            radius = 30
            frame_idx = 30
            while(True):

                if len(ar) - 4 <= frame_idx or len(avg_list) - radius <= frame_idx:
                    break

                if 2 > abs(ar[frame_idx] - ar[frame_idx + 4]) >= threshold:

                    if len(avg_list[frame_idx-radius:frame_idx + radius]) == radius * 2 and len(ar[frame_idx-radius:frame_idx + radius]) == radius * 2:
                        ar_data.append(
                            max(avg_list[frame_idx-radius:frame_idx + radius]))
                        ar_data.append(
                            min(avg_list[frame_idx-radius:frame_idx + radius]))
                        ar_data.append(
                            max(ar[frame_idx-radius:frame_idx + radius]))
                        ar_data.append(
                            min(ar[frame_idx-radius:frame_idx + radius]))
                        fall_data.append(ar_data)

                        if frame_idx-radius <= target_start[num-1] + target_range <= frame_idx + radius and frame_idx-radius <= target_start[num-1] <= frame_idx + radius:
                            fall_target.append(1)
                            print('video:', num)
                            print('picture:', frame_idx)
                            print('label:', 1)
                            frame_idx += radius
                        else:
                            fall_target.append(0)
                            frame_idx += radius

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
            box_top_center_x = []
            box_top_center_y = []
            box_buttom_right_x = []
            box_buttom_right_y = []

    print(len(fall_target))
    print(count)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.externals import joblib

    # fall_data = open(fall_data, newline='')
    # rows = csv.reader(fall_data)
    # fall_target = open(fall_target, newline='')
    # rows = csv.reader(fall_target)

    X = np.array(fall_data)
    Y = np.array(fall_target)
    print(X.shape)
    print(Y.shape)

    svm = joblib.load(clf.pkl)
    y_predict = clf.predict(X)
    print("XOR", Y ^ y_predict)
    plt.scatter(X[:, 0], X[:, 1], c=y_predict)
    plt.savefig("t.png")

    result = Y ^ y_predict
    print("Label", Y)
    print(sum(Y))
    print("Accuracy: {:2f}% {} / {} / {}".format((1 - (result.sum() /
                                                       len(result))) * 100, sum(Y), result.sum(), len(Y)))
