import os
import re
import cv2  # opencv library
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt
from natsort import natsorted

#for j in range(2, 25):
for i in range(21, 222):
    #video_name = "../dataset/data/motion_9/dataset/data_" + str(i) + "/"
    #output_path = "../dataset/data/Diff/motion_9_diff/data_" + str(i) + "/"
    video_name = '../dataset/data/8cam_dataset/chute23/cam6_Trim.mp4'
    output_path = '../dataset/data/8cam_dataset/chute23/cam6_Trim/diff'
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
# 导入视频帧
# get file names of the frames
    #col_frames = natsorted(os.listdir(video_name + "/"))
    # print(col_frames)
    # sort file names
    #col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

    # empty list to store the frames
    col_images = []

    cap = cv2.VideoCapture(r'../dataset/data/8cam_dataset/chute23/cam6_Trim.mp4')
    
    # for i in col_frames:
    while True:
        # read the frames
        # print(i)
        ret, img = cap.read()
        #img = cv2.imread(video_name + "/"+i)
        # append the frames to the list
        if ret == False:
            break
        col_images.append(img)
    i = 0
    for i in range(len(col_images)-1):
        for frame in [i, i+1]:
            cv2.cvtColor(col_images[frame], cv2.COLOR_BGR2GRAY)

        # 像素差值展示
        # convert the frames to grayscale
            grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(col_images[i+1], cv2.COLOR_BGR2GRAY)

        # 图像预处理
            diff_image = cv2.absdiff(grayB, grayA)
            #cv2.imwrite("./Diff/" + video_name + "/"+ str(i+1) +"-diff.jpg",diff_image)
            cv2.imwrite(output_path + "/" + str(i) + ".jpg", diff_image)

# for i in range(len(col_images)-1):

#     # frame differencing
#     grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
#     grayB = cv2.cvtColor(col_images[i+1], cv2.COLOR_BGR2GRAY)
#     diff_image = cv2.absdiff(grayB, grayA)
#     cv2.imwrite("./FDD_data_picture/" + video_name + "/"+ str(i) +"-diff.jpg",diff_image)
