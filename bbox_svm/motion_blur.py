import os
import re
import cv2 # opencv library
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt
from natsort import natsorted

for j in range(2, 25):
    for i in range(1,9) :
        video_name = "./FDD_data_picture/data ("+ str(j) +"_" + str(i) +")" 
        output_path = "Diff/data (" + str(j) + "_" + str(i) +")" + "/"
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
    # 导入视频帧
    # get file names of the frames
        col_frames = natsorted(os.listdir(video_name + "/"))
        #print(col_frames)
        # sort file names
        #col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

        # empty list to store the frames
        col_images=[]

        for i in col_frames:
            # read the frames
            #print(i)
            img = cv2.imread(video_name + "/"+i)
            # append the frames to the list
            col_images.append(img)
        i = 1
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
                cv2.imwrite(output_path + "/" + str(i) +".jpg",diff_image)

# for i in range(len(col_images)-1):

#     # frame differencing
#     grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
#     grayB = cv2.cvtColor(col_images[i+1], cv2.COLOR_BGR2GRAY)
#     diff_image = cv2.absdiff(grayB, grayA)
#     cv2.imwrite("./FDD_data_picture/" + video_name + "/"+ str(i) +"-diff.jpg",diff_image)