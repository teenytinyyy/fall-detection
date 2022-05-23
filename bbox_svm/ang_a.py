'''
def diff(front, ang_list, diff_ang_list, last = 4):
    for front in range(len(ang_list)-last+1):
        if last != 4 :
            diff_ang_list.append(avg(abs(ang_list[front:front+last])))
        diff_ang_list.append(ang_list[front] - ang_list[last])

    return diff_ang_list

ang = []
ang_v = []
ang_a = []
ang_a_avg = []

if __name__ == '__main__' :

    ang_v = diff(0, ang, ang_v)
    ang_a = diff(0, ang_v, ang_a)
    ang_a_avg = diff(0, ang_a, ang_a_avg, 12)

if count >= 4:
# ang(0,4)=ang_v[0]
    ang_v.append(ang[count-4] - ang[count])
if count >= 8:
    ang_a.append(abs(ang_v[count-8]-ang_v[count-4]))
if count >= 19:
    ang_a_avg.append(avg(ang_a[count-19:count-7]))
'''
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

    threshold = 0.6
    angle_list = []
    ar = []
    ar_data = []
    box_top_center_x = []
    box_top_center_y = []
    box_bottom_right_x = []
    box_bottom_right_y = []
    


    for num in range(17,19):
        file_path = "./excel/Home_01/Home_01video (" + str(num) + ").csv"
        # file = open(file_name, newline='')    with用完會幫你關，單用open 要自己關
        # rows = csv.reader(file)
        
        with open(file_path, "r") as r_file:
            rows = csv.reader(r_file)
            for row in rows:
                x_data = [float(val) for val in row]
                ar.append(x_data[0])
                box_top_center_x.append(x_data[1])
                box_top_center_y.append(x_data[2])
                box_bottom_right_x.append(x_data[3])
                box_bottom_right_y.append(x_data[4])
            AR_diff = get_diff_list(ar)
            for i in range(len(box_top_center_x)):
                ang = math.atan2(abs(box_bottom_right_y[i] - box_top_center_y[i]), abs(box_bottom_right_x[i] - box_top_center_x[i]))
                angle_list.append(ang)
            #for j in range(len(AR_diff)):
            #    if AR_diff[i] >= threshold:        
            ang_v_list = get_diff_list(angle_list)
            ang_a_list = get_diff_list(ang_v_list)
            avg_list = get_avg_list(ang_a_list)
            ar_data.append(ar)
            ar = []
            box_top_center_x = []
            box_top_center_y = []
            box_bottom_right_x = []
            box_bottom_right_y = []
    print(ar_data[1])
            #np.savetxt( './excel/fall_data.csv', data, delimiter=',')
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.externals import joblib

    # fall_data = open(fall_data, newline='')
    # rows = csv.reader(fall_data)
    # fall_target = open(fall_target, newline='')
    # rows = csv.reader(fall_target)
    X = ar_data
    #print(X[0])
    #print(X[5])
    Y = fall_target
    # x_train, x_test, y_train, y_test = train_test_split(X, Y,
    #                                                 test_size=0.2,
    #                                                 random_state=87)

    # clf = SVC()
    # clf.fit(x_train, y_train)
    # y_predict = clf.predict(x_test)
