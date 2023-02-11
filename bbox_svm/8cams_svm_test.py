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


def test_svm_classify(data: np.ndarray, target: np.ndarray, check_list: np.ndarray, rev_idx: Dict, train_ratio: float = 0.7):
    randomized_indices = random_indices(0, len(data))
    x_train = data[randomized_indices[0: int(train_ratio * len(data))]][:, 2]
    y_train = target[randomized_indices[0: int(train_ratio * len(data))]]

    x_test = data[randomized_indices[int(train_ratio * len(data)): len(data)]][:, 2]
    print(len(x_test))
    y_test = target[randomized_indices[int(train_ratio * len(data)): len(data)]]
    check_points = check_list[randomized_indices[int(train_ratio * len(data)): len(data)]]

    clf = joblib.load('svm_model.pkl')
    #clf = joblib.load('svm_model_avg_list.pkl')
    #clf = SVC()
    #clf.fit(x_train, y_train)

    y_predict = clf.predict(x_test)
    print(len(y_test), len(y_predict))
    for i in range(len(y_predict)):
        #if y_predict[i] == 1 and check_points[i][0] <= 0 and check_points[i][1] <= 1:
        if y_predict[i] == 1 and check_points[i][1] <= 1:
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

def testing_data(data:list, data_point:list, frame_idx:int, radius:int, step:int = 0):
    data_point.append(max(data[frame_idx - radius + step:frame_idx + radius + step]))
    data_point.append(min(data[frame_idx - radius + step:frame_idx + radius + step]))
    return data_point
    

if __name__ == "__main__":

    target_file = "../dataset/data/excel/8cams_fall_labels.csv"
    delay = "../dataset/data/excel/8cams_delay.csv"
    video_start = 22
    video_end = 23
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

    split_video = []
    for num in range(video_start, video_end + 1):
        for num1 in range(1, 9
        ):
            print(num, num1)
            file_path = "../dataset/data/excel/MOG2_mask_rcnn/gray/mask_rcnn_1320/" + str(num) + "_" + str(num1) + ".csv"
            # file_path = "../dataset/data/excel/2cam/test_data (" + str(num) + ").csv"
            delay_num = int(delay_nums[num - 1][num1 - 1])
                #print(num, num1, delay_num)
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

                    if (x_data[8] - delay_num) <= 0:
                        continue
                    else:                        
                        frame_num.append(x_data[8] - delay_num)
                        ar.append((x_data[3] - x_data[1]) / (x_data[2] - x_data[0]))
                        box_top_center_y.append(x_data[1])                
                        ang = math.atan2(
                            abs((x_data[3] - x_data[1])),
                            abs((x_data[2] - x_data[0]) / 2),
                        )
                        angle_list.append(ang)
                        # frame_num.append(x_data[8] - delay_num)
                        # ar.append((x_data[7] - x_data[5]) / (x_data[6] - x_data[4]))
                        # box_top_center_y.append(x_data[5])                
                        # ang = math.atan2(
                        #     abs((x_data[7] - x_data[5])),
                        #     abs((x_data[6] - x_data[4]) / 2),
                        # )
                        # angle_list.append(ang)

            target_starts = []
            target_ends = []
            target_labels = []
            target_start_1 = -30
            target_end_1 = -30
            target_start_2 = -30
            target_end_2 = -30
            with open(target_file, "r") as r_file:
                rows = csv.reader(r_file)
                for row in rows:
                    target_list = [int(val) for val in row]
                    if target_list[0] == num:
                        target_starts.append(target_list[1])
                        target_ends.append(target_list[2])
                        target_start_1 = target_starts[0]
                        target_end_1 = target_ends[0]

                        if len(target_starts) >= 2:
                            target_start_2 = target_starts[1]
                            target_end_2 = target_ends[1]
                        #print(target_list[0], target_start_1)

            target_idx = 0
            radius = 30
            frame_idx = 30
            ang_v_list = get_diff_list(angle_list)
            ang_a_list = get_diff_list(ang_v_list)
            avg_list = get_abs_avg_list(ang_a_list, avg_range)


            while True:
                    
                # target_end = target_ends[target_idx]
                # if frame_num[frame_idx] > target_end:
                #     target_idx += 1
                # if target_idx == len(target_starts):
                #     break                
                # target_start = target_starts[target_idx]
                # target_end = target_ends[target_idx]
                # target_label = target_labels[target_idx]

                if len(ar) - 4 <= frame_idx or len(avg_list) - radius <= frame_idx:
                    break

                if 2 > abs(ar[frame_idx] - ar[frame_idx + 4]) >= threshold:
                #if 0.8 > (ar[frame_idx]) >= threshold:
                #if avg_list[frame_idx] >= threshold*0.1:
                    #print(num, num1, frame_num[frame_idx], target_start_1 - delay_num)

                    if (len(avg_list[frame_idx - radius: frame_idx + radius]) == radius * 2 and len(ar[frame_idx - radius: frame_idx + radius]) == radius * 2):
                        # if target_end - 10 <= frame_num[frame_idx + radius] <= target_end + radius:
                        #     split_video.append((num, num1, frame_num[frame_idx - radius], frame_num[frame_idx + radius], target_start, target_end, target_label))
                        # frame_idx += radius
                        # ar_data.append(
                        #     max(avg_list[frame_idx - radius:frame_idx + radius]))
                        # ar_data.append(
                        #     min(avg_list[frame_idx - radius:frame_idx + radius]))
                        ar_data = []
                        batch_data = []
                        check_points = []
                        testing_data(avg_list, ar_data, frame_idx, radius, 0)
                        testing_data(ar, ar_data, frame_idx, radius, 0)
                        #print(ar_data)
                        batch_data.append(ar_data)
                        ar_data = []
                        testing_data(avg_list, ar_data, frame_idx, radius, 10)
                        testing_data(ar, ar_data, frame_idx, radius, 10)
                        
                        batch_data.append(ar_data)                            
                        ar_data = []
                        testing_data(avg_list, ar_data, frame_idx, radius, 15)
                        testing_data(ar, ar_data, frame_idx, radius, 15)
                        
                        batch_data.append(ar_data)   

                        
                        check_points.append(box_top_center_y[frame_idx - radius] - box_top_center_y[frame_idx + radius + 20])
                        check_points.append(ar[frame_idx + radius + 10])

                        fall_data.append(batch_data)
                        check_list.append(check_points)



                        # if target_label == 2:
                        # if (frame_idx - radius <= target_start[num - 1] + target_range <= frame_idx + radius):
                        if (frame_num[frame_idx - radius] <= target_start_1 + target_range <= frame_num[frame_idx + radius]):
                            # else:                
                            #print(frame_num[frame_idx + radius], target_start_1 + target_range)                
                            fall_target.append(1)
                            #print(1)
                            print("video:", num, num1)
                            print("picture:", frame_num[frame_idx])
                            print("label:", 1)
                            #print(ar_data)

                            frame_idx += radius
                        elif (frame_num[frame_idx - radius] <= target_end_1 - target_range <= frame_num[frame_idx + radius]):
                            print(2)
                            fall_target.append(1)
                            print("video:", num, num1)
                            print("picture:", frame_num[frame_idx])
                            print("label:", 1)
                            frame_idx += radius
                        elif (frame_num[frame_idx - radius] <= target_start_2 + target_range <= frame_num[frame_idx + radius]):
                            print(3)
                            fall_target.append(1)
                            print("video:", num, num1)
                            print("picture:", frame_num[frame_idx])
                            print("label:", 1)

                            frame_idx += radius
                        elif (frame_num[frame_idx - radius] <= target_end_2 - target_range <= frame_num[frame_idx + radius]):
                            fall_target.append(1)
                            print("video:", num, num1)
                            print("picture:", frame_num[frame_idx])
                            print("label:", 1)       
                            frame_idx += radius                 
                        else:
                            #if not_fall % 2 == 0:
                            fall_target.append(0)
                            frame_idx += radius
                            #not_fall += 1

                        rev_idx[len(fall_target) - 1] = (
                            num,
                            num1,
                            frame_num[frame_idx - 2 * radius] + delay_num,
                            frame_num[frame_idx] + delay_num,
                            fall_target[-1],
                        )


                    count += 1
                    frame_idx += 1

                    continue


                frame_idx += 1
        # np.savetxt("../dataset/data/excel/mask_rcnn_1286/1286target_labeltest4.csv", np.c_[split_video], delimiter=",")


    print("fall_target", len(fall_target))
    print("thres", count)
    print("fall", sum(fall_target))
    # print("count_2", count_2)

    X_1 = np.array(fall_data)
    Y = np.array(fall_target)
    check_list = np.array(check_list)


    test_svm_classify(X_1, Y, check_list, rev_idx, 0)