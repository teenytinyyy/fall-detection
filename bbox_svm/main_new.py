# coding=utf-8

# 加载一些基础包以及设置logger
from audioop import avg
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
for num in range(17, 23):
    checkpoint_threshold = 0.7
    histogram_tresh = 0.3  # 20%閥值
    test_mode = False
    h_thresh = True  # control testing true or false(test bool)
    svm_model = "true"  # control svm model true or false(data bool)
    video_name = "Home_01video (" + str(num) + ")"
    video_frame = 8
    #class_name = "fall"

    if h_thresh:
        test_bool = "true"
    else:
        test_bool = "false"
    output_path = "frame"+"data_" + svm_model + \
        "/" + "test_"+test_bool + video_name + "/"
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    if __name__ == '__main__':
        # output_path = "result_instance_segmentation.jpg"
        # for i in range(83):      #循环次数自己选择
        #   image=cv2.imread("/home/ian/code/detectron2_repo/demo/val2017/%d.jpg"%(i+1))#路径自己选择
        #   image.convert("RGB")

        images = [cv2.imread(file) for file in natsorted(
            glob.glob("./FDD_data_picture/" + video_name + "/*.jpg"))]

        #cap = cv2.VideoCapture(input_path)
        # 指定模型的配置配置文件路径及网络参数文件的路径
        # 对于像下面这样写法的网络参数文件路径，程序在运行的时候就自动寻找，如果没有则下载。
        # Instance segmentation model

        model_file_path = './detectron_model/config.yaml'

        model_weights = "./detectron_model/model_final.pth"

        svm_model_weights = "./SVM_model/svm_true_fall101_new_v2_c3095.model"

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

        offset = (0, 0)

        ang = []
        ang_v = []
        ang_a = []
        ang_a_avg = []
        box_top = []
        box_buttom = []
        bbox_l = []
        bbox_w = []
        bbox_r = []
        bbox_all = []
        bbox_data = []
        frame_num = []
        center_dis = []
        box_center_x = []
        box_center_y = []
        box_top_center = []
        box_buttom_right = []
        top_center_data = []
        c = 0
        count = -1
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
                single_prediction_cpu = single_prediction["instances"].to(
                    "cpu")._fields
            person_selection_mask = single_prediction_cpu["pred_classes"] == 0
            box_selection_mask = person_selection_mask
            predictions.append({"pred_boxes": single_prediction_cpu["pred_boxes"].tensor[box_selection_mask].data.numpy(),
                                "scores": single_prediction_cpu["scores"][box_selection_mask].data.numpy(),
                                "pred_masks": single_prediction_cpu["pred_masks"][box_selection_mask].data.numpy()})

            if len(predictions[0]["pred_masks"]) != 0:  # 防呆 is not
                array = predictions[0]["pred_masks"][0]  # 第一個人的MASK拿出來，只測一人

                count += 1
                print(count)

                polygons = Mask(array).polygons()
                m_x1, m_y1, m_x2, m_y2 = polygons.bbox()  # maskrcnn的bbox
                #binarizedImage = (predictions[0]["pred_masks"][0]  > 126) * 255
                binarizedImage = predictions[0]["pred_masks"][0]

                horizontal_projection = np.sum(binarizedImage, axis=0)

                y1_max = np.max(horizontal_projection)  # 鉛直投影
                thresh_ = y1_max * histogram_tresh  # 1全部加起來最高的

                for i in range(len(horizontal_projection)):
                    if horizontal_projection[i] >= thresh_:

                        h_x1 = i
                        break
                for j in range(len(horizontal_projection)-1, 0, -1):
                    if horizontal_projection[j] >= thresh_:

                        h_x2 = j
                        break
                box_r_ori = ((m_y2-m_y1)/(h_x2-h_x1))
                if box_r_ori <= 1:  # 斜躺才修正

                    new_m_y1 = m_y2 - y1_max
                else:
                    new_m_y1 = m_y1

                m_cX = int((h_x1 + h_x2) / 2.0)
                m_cY = int((new_m_y1 + m_y2) / 2.0)

                cv2.rectangle(frame, (h_x1, new_m_y1),
                              (h_x2, m_y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (m_x1, m_y1),
                              (m_x2, m_y2), (0, 0, 255), 3)

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

            start = time.time()

            if len(predictions[0]["pred_masks"]) != 0:
                box_center_x.append(m_cX)
                box_center_y.append(m_cY)
                box_top_center.append((m_cX, m_y1))
                box_buttom_right.append((m_x2, m_y2))
                bbox_r.append((m_y2-new_m_y1)/(h_x2-h_x1))
                ang.append(math.atan2(abs(m_y2 - m_y1),
                                      abs(m_cX-m_x2)))  # (y,x)
                if count >= 4:
                    # ang(0,4)=ang_v[0]
                    ang_v.append(ang[count-4] - ang[count])
                if count >= 8:
                    ang_a.append(abs(ang_v[count-8]-ang_v[count-4]))
                if count >= 19:
                    ang_a_avg.append(avg(ang_a[count-19:count-7]))

                if flag_checkpoint == True:
                    # Check moving

                    if (b+1) % video_frame == 0:
                        dis_label = b

                    box_center_dis = eucliDist((box_center_x[count-video_frame*2] / frame_width, box_center_y[count-video_frame*2] /
                                                frame_height), (box_center_x[count] / frame_width, box_center_y[count] / frame_height))

                    if (b+1) % video_frame == 0:
                        last_5_box_r = (
                            bbox_r[count] + bbox_r[count-1] + bbox_r[count-2] + bbox_r[count-3] + bbox_r[count-4])/5
                        #last_5_center_dis = (center_dis[0] + center_dis[1] + center_dis[2] + center_dis[3] + center_dis[4])/5
                        last_5_center_dis = (box_center_dis)
                        bbox_data.append(
                            bbox_r[range(count-video_frame*2, count+1)])
                        top_center_data.append(
                            box_top_center[range(count-video_frame*2, count+1)])

                        front_5_frame_r = (bbox_r[count-video_frame*2] + bbox_r[count-video_frame*2+1] + bbox_r[count -
                                                                                                                video_frame*2+2] + bbox_r[count-video_frame*2+3] + bbox_r[count-video_frame*2+4])/5
                        data = np.array(
                            [front_5_frame_r, last_5_box_r, last_5_center_dis]).reshape(1, -1)
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
                        #np.savetxt( video_name + '.csv', np.c_[label],delimiter=',')
                        if flag2 == True:
                            frame = cv2.putText(frame, "Warning", (int(
                                frame_width/20), int(frame_height / 5)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                        if test_mode:
                            frame = cv2.putText(frame,  action, (int(
                                frame_width/30), int(frame_height / 20)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                            cv2.imwrite(output_path + "%d.jpg" % (c), frame)
                    b += 1
                    print("b:", b)
                    # Check moving End

                    if b == video_frame:
                        flag_checkpoint = False
                        b = 0
                        ang_v = []

                else:
                    if count >= video_frame*2 and 2 > abs(bbox_r[count - 4] - bbox_r[count]) > checkpoint_threshold:
                        print("bbox_r[c]:", bbox_r[count])
                        print("bbox_r[c-4]:", bbox_r[count - 4])
                        flag_checkpoint = True

            # Calculate End

            if flag2 == True:
                frame = cv2.putText(frame, "Warning", (int(
                    frame_width/20), int(frame_height / 5)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
            if test_mode is False:
                if action is not None:
                    if action == "fall down":
                        frame = cv2.putText(frame,  action, (int(
                            frame_width/30), int(frame_height / 20)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 250))
                        cv2.imwrite(output_path + "%d.jpg" % (c), frame)
                    else:
                        frame = cv2.putText(frame,  action, (int(
                            frame_width/30), int(frame_height / 20)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 180, 0))
                        cv2.imwrite(output_path + "%d.jpg" % (c), frame)
                else:
                    cv2.imwrite(output_path + "%d.jpg" % (c), frame)

            c = c+1
            print("c:", c)
            end = time.time()

            cv2.destroyAllWindows()
            np.savetxt('./excel/Home_01/' + video_name + '.csv',
                       np.c_[bbox_r, box_top_center, box_buttom_right], delimiter=',')
