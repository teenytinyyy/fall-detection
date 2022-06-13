import csv
import math
import numpy as np
from typing import List
import random


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
        avg = np.average(np.absolute(np_angle_list[idx:idx + avg_range]))
        avg_list.append(avg)

    return avg_list


def get_avg_list(angle_list: List[float], avg_range: int = 13) -> List[float]:

    np_angle_list = np.array(angle_list)

    avg_list = []
    for idx in range(len(np_angle_list) - avg_range):
        #avg = np.average(np.absolute(np_angle_list[idx:idx + avg_range]))
        avg = np.average(np_angle_list[idx:idx + avg_range])
        avg_list.append(avg)

    return avg_list


if __name__ == '__main__':

    target = "../dataset/data/excel/2cam/svm_2cam_target.csv"
    unused = "../dataset/data/excel/2cam/2cam.csv"
    video_start = 1
    video_end = 221
    target_range = 13
    threshold = 0.6
    avg_range = 13
    angle_list = []
    ar = []
    ar_data = []
    box_top_center_x = []
    box_top_center_y = []
    box_bottom_right_x = []
    box_bottom_right_y = []
    fall_target = []
    fall_data = []
    labels = []
    max_data = []
    min_data = []
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
        file_path = "../dataset/data/excel/2cam/data (" + str(num) + ").csv"
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
                ang = math.atan2(abs(box_bottom_right_y[i] - box_top_center_y[i]), abs(
                    box_bottom_right_x[i] - box_top_center_x[i]))
                angle_list.append(ang)
                #y_dis = box_top_center_y[i] - box_bottom_right_y[i]
                #y_dis_list.append(y_dis)

            #y_v_list = get_diff_list(y_dis_list)
            #y_a_list = get_diff_list(y_v_list)
            #y_a_avg_list = get_avg_list(y_a_list, 13)
            y_top_dis = get_diff_list(box_top_center_y)
            y_top_avg = get_avg_list(y_top_dis, avg_range)
            ang_v_list = get_diff_list(angle_list)
            ang_a_list = get_diff_list(ang_v_list)
            avg_list = get_abs_avg_list(ang_a_list, avg_range)

            # for frame_idx in range(len(avg_list)-30):
            #    print(frame_idx, max(avg_list[frame_idx:frame_idx + radius]) / (min(avg_list[frame_idx:frame_idx + radius]) + 1e-8))

            radius = 30
            frame_idx = 30
            while(True):

                # print(frame_idx)
                if len(ar) - 4 <= frame_idx or len(avg_list) - radius <= frame_idx:
                    break

                if 2 > abs(ar[frame_idx] - ar[frame_idx + 4]) >= threshold:

                    #fall_data.append(avg_list[frame_idx-radius:frame_idx + radius])
                    # print(len(avg_list[frame_idx-radius:frame_idx + radius]))
                    #fall_data.append(ar[frame_idx-radius:frame_idx + radius])
                    # print(len(ar[frame_idx-radius:frame_idx + radius]))
                    if len(avg_list[frame_idx - radius:frame_idx + radius]) == radius * 2 and len(ar[frame_idx - radius:frame_idx + radius]) == radius * 2:
                        # ar_data.append(fall_data)
                        ar_data.append(
                            max(avg_list[frame_idx - radius:frame_idx + radius]))
                        # ar_data.append(
                        #     min(avg_list[frame_idx - radius:frame_idx + radius]))
                        ar_data.append(
                            max(ar[frame_idx - radius:frame_idx + radius]))
                        ar_data.append(
                            min(ar[frame_idx - radius:frame_idx + radius]))
                        # ar_data.append(
                        #     max(y_top_avg[frame_idx - radius:frame_idx + radius]))   
                        # ar_data.append(
                        #     min(y_top_avg[frame_idx - radius:frame_idx + radius]))                      
                        # ar_data.append(
                        #     min(y_a_avg_list[frame_idx - radius:frame_idx + radius]))
                                                                               
                        fall_data.append(ar_data)
                        # for plus in range(radius):
                        # ar_data.append(avg_list[frame_idx-radius:frame_idx + radius] + ar[frame_idx-radius:frame_idx + radius])

                        # if frame_idx - radius <= target_start[num - 1] + target_range <= frame_idx + radius and frame_idx - radius <= target_start[num - 1] <= frame_idx + radius:
                        if frame_idx - radius <= target_start[num - 1] + target_range <= frame_idx + radius:
                            fall_target.append(1)
                            print('video:', num)
                            print('picture:', frame_idx)
                            print('label:', 1)
                            # print(min(y_a_avg_list[frame_idx - radius:frame_idx + radius]))
                            print(min(y_top_avg[frame_idx - radius:frame_idx + radius]))
                            print(max(y_top_avg[frame_idx - radius:frame_idx + radius]))

                            frame_idx += radius
                        else:
                            # print(0)
                            fall_target.append(0)
                            frame_idx += radius

                        rev_idx[len(fall_target) - 1] = (num,
                                                         frame_idx - radius, fall_target[-1])
                        # frame_idx += radius

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
            box_bottom_right_x = []
            box_bottom_right_y = []
            y_dis_list = []
            y_v_list = []
            y_a_list = []

    print(len(fall_target))
    print(count)

    # print(ar_data[1])
    #np.savetxt( './excel/fall_data.csv', data, delimiter=',')

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    # from sklearn.externals import joblib

    # fall_data = open(fall_data, newline='')
    # rows = csv.reader(fall_data)
    # fall_target = open(fall_target, newline='')
    # rows = csv.reader(fall_target)

    X = np.array(fall_data)
    Y = np.array(fall_target)

    random_idx = random_indices(0, len(X))
    x_train = X[random_idx[0:int(0.7 * len(X))]]
    y_train = Y[random_idx[0:int(0.7 * len(X))]]
    x_test = X[random_idx[int(0.7 * len(X)):len(X)]]
    y_test = Y[random_idx[int(0.7 * len(X)):len(X)]]

    clf = SVC()
    clf.fit(x_train, y_train)
    # joblib.dump(clf, '../states/2cam.pkl')

    y_predict = clf.predict(x_test)
    diff_idx = np.where(y_test ^ y_predict > 0)[0]

    print("diff indices:")
    print(diff_idx)

    diff_dict = {}
    for i in diff_idx:
        restored_idx = random_idx[int(0.7 * len(X)):len(X)][i]
        diff_dict[restored_idx] = [restored_idx, y_predict[i], y_test[i]]
        print(rev_idx[restored_idx], diff_dict[restored_idx])

        try:
            assert rev_idx[restored_idx][2] == diff_dict[restored_idx][2]

        except AssertionError:
            print("ERROR gt not match")

    print("Diff count:{}".format(len(diff_dict.keys())))

    idx_list = np.array(
        random_idx[int(0.7 * len(X)):len(X)])[diff_idx].astype(np.uint8)

    # print("source info")
    # for i in idx_list:
    #     print(i, rev_idx[i])

    # print("XOR", y_test^y_predict)
    # plt.scatter(x_test[:,0], x_test[:,1], c=y_predict)
    # plt.savefig("t.png")

    # for k, v in rev_idx.items():
    #     print("rev", k, v)

    confusion_mat = np.zeros((2, 2))
    idx_pairs = np.array([y_predict.tolist(), y_test.tolist()]).T.tolist()
    a = 0
    for idx_pair in idx_pairs:
        # print("gt", Y[random_idx[int(0.7 * len(X)):len(X)][a]], y_test[a])
        confusion_mat[tuple(idx_pair)] += 1
        a += 1

    print("confusion matrix")
    print(confusion_mat)
    result = y_test ^ y_predict

    print(sum(Y))
    print("Accuracy: {:2f}% {} / {} / {}".format((1 - (result.sum() /
          len(result))) * 100, sum(y_test), result.sum(), len(y_test)))
