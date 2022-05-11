# coding=utf-8

# 加载一些基础包以及设置logger
from sklearn.svm import SVC, LinearSVC
from natsort import natsorted
import joblib
from imantics import Polygons, Mask
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import os
import time
import cv2
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger
import csv
import math
import glob
setup_logger()

# 加载其它一些库
# 加载相关工具
#input_path = "/home/ian/code/falldata/Office/office(1).avi"

# input_path = "/home/ian/code/detectron2_repo/demo/bbox_middle/fall1"
# out_put_img = "/home/ian/code/detectron2_repo/demo/bbox_middle/fall1/coffee(1)047.jpg"
# 圖片轉影片指令 ffmpeg -i %1d.jpg -r 25 output.mp4
# 影片轉圖片指令 ffmpeg -i input.mkv out%d.bmp

checkpoint_threshhold = 0.7
histogram_tresh = 0.3  # 20%閥值
test_mode = False
h_thresh = True  # control testing true or false(test bool)
svm_model = "true"  # control svm model true or false(data bool)
video_name = "Home_01video (3)"
video_frame = 8
#class_name = "fall"


if h_thresh:
    test_bool = "true"
else:
    test_bool = "false"
output_path = "frame"+"data_" + svm_model + "/" + "test_"+test_bool + video_name + "/"
if not os.path.isdir(output_path):
    os.makedirs(output_path)


# def key_func():
#      return os.path.split("1")[-1]

if __name__ == '__main__':
    # for i in range(83):      #循环次数自己选择
    images = [cv2.imread(file) for file in natsorted(glob.glob("./FDD_data_picture/" + video_name + "/*.jpg"))]

    # print(images[0])
    #cap = cv2.VideoCapture(input_path)
    # 指定模型的配置配置文件路径及网络参数文件的路径
    # 对于像下面这样写法的网络参数文件路径，程序在运行的时候就自动寻找，如果没有则下载。
    model_file_path = './detectron_model/config.yaml'
    model_weights = "./detectron_model/model_final.pth"
    svm_model_weights = "./SVM_model/svm_true_fall101_new_v2_c3095.model"

    # 加载图片
    # img = cv2.imread(input_path)

    # 创建一个detectron2配置
    cfg = get_cfg()
    # 要创建模型的名称
    cfg.merge_from_file(model_file_path)
    # 为模型设置阈值
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    #cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.95
    # 加载模型需要的数据
    cfg.MODEL.WEIGHTS = model_weights
    # 基于配置创建一个默认推断
    predictor = DefaultPredictor(cfg)

    svm = joblib.load(svm_model_weights)

    # 利用这个推断对加载的影像进行分析并得到结果
    # 对于输出结果格式可以参考这里https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #out = cv2.VideoWriter('/home/ian/code/detectron2_repo/demo/output999.mp4',fourcc, 25.0, (320,  240))
    offset = (0, 0)

    box_top = []
    box_buttom = []
    bbox_l = []
    bbox_w = []
    bbox_r = []
    bbox_all = []
    frame_num = []
    center_dis = []
    box_center_x = []
    box_center_y = []
    c = 0
    b = 0
    dis_label = 0
    box_r = 0
    box_center_dis = 0
    box_center_dis_prev = 0
    action = None
    flag = False
    flag2 = False
    flag_checkpoint = False

    def eucliDist(A, B):
        return math.sqrt(sum([(a - b)**2 for (a, b) in zip(A, B)]))

    print("images#:", len(images))
    for idx, frame in enumerate(images):
        # print(ret)
        print(idx)

        outputs = predictor(frame)
        # print(outputs)

        predictions = []
        for single_prediction in outputs:
            # Transfer relevant data to cpu
            single_prediction = outputs
            # print(single_prediction)
            single_prediction_cpu = single_prediction["instances"].to("cpu")._fields
        person_selection_mask = single_prediction_cpu["pred_classes"] == 0
        box_selection_mask = person_selection_mask
        predictions.append({"pred_boxes": single_prediction_cpu["pred_boxes"].tensor[box_selection_mask].data.numpy(),
                            "scores": single_prediction_cpu["scores"][box_selection_mask].data.numpy(),
                            "pred_masks": single_prediction_cpu["pred_masks"][box_selection_mask].data.numpy()})

        if len(predictions[0]["pred_masks"]) != 0:  # 防呆 is not
            array = predictions[0]["pred_masks"][0]  # 第一個人的MASK拿出來，只測一人

            polygons = Mask(array).polygons()
            m_x1, m_y1, m_x2, m_y2 = polygons.bbox()  # maskrcnn的bbox
            #binarizedImage = (predictions[0]["pred_masks"][0]  > 126) * 255
            binarizedImage = predictions[0]["pred_masks"][0]

            # print(binarizedImage)
            horizontal_projection = np.sum(binarizedImage, axis=0)
            # print(m_x1,m_y1,m_x2,m_y2)
            # print(horizontal_projection)
            y1_max = np.max(horizontal_projection)  # 鉛直投影
            thresh_ = y1_max * histogram_tresh  # 1全部加起來最高的
            # print(y1_max)
            for i in range(len(horizontal_projection)):
                if horizontal_projection[i] >= thresh_:
                    # print(horizontal_projection[i],i)
                    h_x1 = i
                    break
            for j in range(len(horizontal_projection)-1, 0, -1):
                if horizontal_projection[j] >= thresh_:
                    # print(horizontal_projection[j],j)
                    h_x2 = j
                    break
            box_r_ori = ((m_y2-m_y1)/(h_x2-h_x1))
            if box_r_ori <= 1:  # 斜躺才修正

                new_m_y1 = m_y2 - y1_max
            else:
                new_m_y1 = m_y1


            m_cX = int((h_x1 + h_x2) / 2.0)
            m_cY = int((new_m_y1 + m_y2) / 2.0)
            cv2.rectangle(frame, (m_x1, m_y1), (m_x2, m_y2), (0, 0, 255), 3)
           

        bbox = predictions[0]["pred_boxes"]

        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
        frame_num.append(c+1)


        frame_width = int(frame.shape[1])
        frame_height = int(frame.shape[0])
        print("frame_width = ", frame_width)
        print("frame_height = ", frame_height)


        start = time.time()

        # Calculate start
        if len(predictions[0]["pred_masks"]) != 0:
            box_center_x.append(m_cX)
            box_center_y.append(m_cY)
            bbox_r.append((m_y2-new_m_y1)/(h_x2-h_x1))
            print(bbox_r)

            if flag_checkpoint == True:
                # Check moving

                if (b+1) % video_frame == 0:
                    dis_label = b

                box_center_dis = eucliDist((box_center_x[dis_label] / frame_width, box_center_y[dis_label] / frame_height), (m_cX / frame_width, m_cY / frame_height))
                #box_center_dis_prev = eucliDist((box_center_x[dis_label-1] / frame_width,box_center_y[dis_label-1]),(m_cX / frame_width,m_cY / frame_height))
                center_dis.append(box_center_dis)
                print(box_center_x[dis_label])
                print(box_center_y[dis_label])
                print(center_dis)

                if (b+1) % video_frame == 0:
                    last_5_box_r = (bbox_r[c] + bbox_r[c-1] + bbox_r[c-2] + bbox_r[c-3] + bbox_r[c-4])/5
                    last_5_center_dis = (center_dis[0] + center_dis[1] + center_dis[2] + center_dis[3] + center_dis[4])/5
                    front_5_frame_r = (bbox_r[c-video_frame*2] + bbox_r[c-video_frame*2+1] + bbox_r[c-video_frame*2+2] + bbox_r[c-video_frame*2+3] + bbox_r[c-video_frame*2+4])/5
                    data = np.array([front_5_frame_r, last_5_box_r, last_5_center_dis]).reshape(1, -1)
                    label = svm.predict(data)
                    print("label:", label)
                    if label == 5:
                        action = "fall down"
                        flag = True
                    if label == 4:
                        action = "lie"
                        if flag == True:
                            flag2 = True
                    if label == 3:
                        action = "sit down/sitting"
                    if label == 2:
                        action = "stand up"
                    if label == 1:
                        action = "walking/standing"

                    print("front_5_frame_r:", front_5_frame_r)
                    print("last_5_box_r:", last_5_box_r)
                    print("last_5_center_dis:", last_5_center_dis)
                    print(action)
                    if flag2 == True:
                        frame = cv2.putText(frame, "Warning", (int(frame_width/20), int(frame_height / 5)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                    if test_mode:
                        frame = cv2.putText(frame,  action  , (int(frame_width/30), int(frame_height / 20)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                        cv2.imwrite(output_path + "%d.jpg" % (c), frame)
                b += 1
                print(b)
                # Check moving End

                if b == video_frame:
                    flag_checkpoint = False
                    b = 0

            else:
                if c >= 4 and 2 > abs(bbox_r[c - 4] - bbox_r[c]) > checkpoint_threshhold:
                    print(bbox_r[c])
                    print(bbox_r[c - 4])
                    flag_checkpoint = True

        # Calculate End

        if flag2 == True:
            frame = cv2.putText(frame, "Warning", (int(frame_width/20), int(frame_height / 5)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
        if test_mode is False:
            if action is not None:
                if action == "fall down":
                    frame = cv2.putText(frame,  action , (int(frame_width/30), int(frame_height / 20)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 250))
                    cv2.imwrite(output_path + "%d.jpg" % (c), frame)
                else:
                    frame = cv2.putText(frame,  action , (int(frame_width/30), int(frame_height / 20)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 180, 0))
                    cv2.imwrite(output_path + "%d.jpg" % (c), frame)
            else:
                cv2.imwrite(output_path + "%d.jpg" % (c), frame)

        c = c+1
        print(c)
        end = time.time()
        print("執行時間：%f 秒" % (end - start))

        # 将影像保存到文件

        cv2.destroyAllWindows()

    print(svm_model_weights)
