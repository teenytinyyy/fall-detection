# coding=utf-8

# 加载一些基础包以及设置logger
from re import T
from sklearn.svm import SVC, LinearSVC
from natsort import natsorted
import joblib
from imantics import Mask
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import os
import time
import cv2
import numpy as np
from detectron2.utils.logger import setup_logger
import csv
import math
import glob
import label as label_utils
import image as image


setup_logger()

if __name__ == "__main__":

    # for num in range(1, 2):
    for num1 in range(1, 222):
        checkpoint_threshold = 0.6
        histogram_thresh = 0.3  # 20%閥值
        test_mode = False
        h_thresh = True  # control testing true or false(test bool)
        svm_model = "true"  # control svm model true or false(data bool)
        video_name = "data (" + str(num1) + ")"
        video_frame = 8
        # class_name = "fall"

        if h_thresh:
            test_bool = "true"
        else:
            test_bool = "false"
        output_path = "../dataset/data/Retinanet/ellipse/" + video_name + "/"
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        # def key_func():
        #      return os.path.split("1")[-1]

        images = [
            cv2.imread(file)
            for file in natsorted(
                glob.glob("../dataset/data/FDD_data_picture/" +
                          video_name + "/*.jpg")
            )
        ]

        model_file_path = "../states/detectron_model/config.yaml"
        model_weights = "../states/detectron_model/model_final.pth"
        svm_model_weights = "../states/SVM_model/svm_true_fall101_new_v2_c3095.model"

        # 创建一个detectron2配置
        cfg = get_cfg()
        # 要创建模型的名称
        cfg.merge_from_file(model_file_path)
        # 为模型设置阈值
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.95
        # 加载模型需要的数据
        cfg.MODEL.WEIGHTS = model_weights
        # 基于配置创建一个默认推断
        predictor = DefaultPredictor(cfg)

        svm = joblib.load(svm_model_weights)

        offset = (0, 0)

        box_top = []
        box_bottom = []
        bbox_l = []
        bbox_w = []
        bbox_r = []
        bbox_all = []
        frame_num = []
        center_dis = []
        box_len_x = []
        box_len_y = []
        box_top_y = []
        ellipse_width = []
        ellipse_height = []
        ellipse_angle = []
        img_list = []

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

        m_x1_p1 = 0
        m_y1_p1 = 0
        m_x2_p1 = 0
        m_y2_p1 = 0

        def eucliDist(A, B):
            return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))

        print("images#:", len(images))
        for idx, frame in enumerate(images):
            outputs = predictor(frame)

            predictions = []

            for single_prediction in outputs:
                # Transfer relevant data to cpu
                single_prediction = outputs
                # print(single_prediction)
                single_prediction_cpu = single_prediction["instances"].to(
                    "cpu")._fields

            person_selection_mask = single_prediction_cpu["pred_classes"] == 0
            box_selection_mask = person_selection_mask
            predictions.append(
                {
                    "pred_boxes": single_prediction_cpu["pred_boxes"]
                    .tensor[box_selection_mask]
                    .data.numpy(),
                    "scores": single_prediction_cpu["scores"][
                        box_selection_mask
                    ].data.numpy(),
                    "pred_masks": single_prediction_cpu["pred_masks"][
                        box_selection_mask
                    ].data.numpy(),
                }
            )

            if len(predictions[0]["pred_masks"]) != 0:  # 防呆 is not
                # 第一個人的MASK拿出來，只測一人
                mask = predictions[0]["pred_masks"][0]
                if len(predictions[0]["pred_masks"]) >= 2:
                    array_p1 = predictions[0]["pred_masks"][1]
                    polygons_p1 = Mask(array_p1).polygons()
                    m_x1_p1, m_y1_p1, m_x2_p1, m_y2_p1 = polygons_p1.bbox()

                count += 1

                angle_1, width_1, height_1, center_1, points_1 = label_utils.mask2ellipse(mask)

                # m_x1, m_y1, m_x2, m_y2 = polygons.bbox()  # maskrcnn的bbox
                # binarized_img = (predictions[0]["pred_masks"][0]  > 126) * 255
                binarized_img = predictions[0]["pred_masks"][0]
                horizontal_projection = np.sum(binarized_img, axis=0)
                y1_max = np.max(horizontal_projection)  # 鉛直投影
                thresh_ = y1_max * histogram_thresh  # 1全部加起來最高的

                for i in range(len(horizontal_projection)):
                    if horizontal_projection[i] >= thresh_:
                        for j in range(len(binarized_img)):                            
                            binarized_img[j][0 : i] = False
                        break

                for i in range(len(horizontal_projection) - 1, 0, -1):
                    if horizontal_projection[i] >= thresh_:
                        for j in range(len(binarized_img)):                            
                            binarized_img[j][i : -1] = False
                        break

                angle_2, width_2, height_2, center_2, points_2 = label_utils.mask2ellipse(binarized_img)

                # cv2.circle(frame, (m_cX, m_cY), 4, (0, 255, 0), -1)
                # cv2.rectangle(frame, (h_x1, new_m_y1),
                #                (h_x2, m_y2), (0, 255, 0), 4)
                # cv2.rectangle(frame, (m_x1, m_y1),
                #               (m_x2, m_y2), (0, 255, 0), 3)
                cv2.polylines(frame, pts=[points_1], isClosed=True, color=(
                    0, 255, 0), thickness=2)
                cv2.polylines(frame, pts=[points_2], isClosed=True, color=(
                    0, 0, 255), thickness=3)
                cv2.ellipse(frame, (int(center_1[1]), int(center_1[0])), (int(
                    height_1), int(width_1)), int(-angle_1), 0, 360, (255, 0, 0), 4)
                cv2.ellipse(frame, (int(center_2[1]), int(center_2[0])), (int(
                    height_2), int(width_2)), int(-angle_2), 0, 360, (255, 255, 0), 2)
                # cv2.rectangle(frame, (m_x1_p1, m_y1_p1),
                #               (m_x2_p1, m_y2_p1), (0, 0, 255), 2)
                # frame = Mask(binarized_img).draw(frame, color=(255, 0, 255), alpha=0.5)

            bbox = predictions[0]["pred_boxes"]

            for i, box in enumerate(bbox):
                x1, y1, x2, y2 = [int(i) for i in box]
                x1 += offset[0]
                x2 += offset[0]
                y1 += offset[1]
                y2 += offset[1]
                cX = int((x1 + x2) / 2.0)
                cY = int((y1 + y2) / 2.0)

            frame_width = int(frame.shape[1])
            frame_height = int(frame.shape[0])
            # print("frame_width = ", frame_width)
            # print("frame_height = ", frame_height)
            # if c > 0:
            #      box_center_dis = eucliDist((box_len_x[0] / frame_width ,box_top_y[0] / frame_height),(m_cX / frame_width,m_cY / frame_height))
            #      box_center_dis_prev = eucliDist((box_len_x[c-1] / frame_width,box_top_y[c-1]),(m_cX / frame_width,m_cY / frame_height))
            #      center_dis.append(box_center_dis)

            # writer.writerow([box_r,cX,cY,box_center_dis,box_center_dis_prev])
            # cv2.imshow('ddd',frame)
            # cv2.imwrite(output_path+"%d.jpg"%(c) , frame)
            # cv2.imwrite(output_path +"%d.jpg"%(c) , result)
            # print(output_path +"%d.jpg"%(c))

            start = time.time()
            # print(c+1)
            # print(bbox)

            # print(cX,cY)
            # if c > 0:
            #      print(box_len_x[c-1] , '-' , cX , '+' , box_top_y[c-1] , '-' , cY,'=' ,box_center_dis)

            # Calculate start
            if len(predictions[0]["pred_masks"]) != 0:
                if height_2 > width_2:
                    angle_2 = 90 - angle_2
                    width_2, height_2 = height_2, width_2
                if angle_2 < 45:
                    len_x_2 = math.cos(angle_2 * math.pi/180) * height_2 * 2
                    len_y_2 = math.cos(angle_2 * math.pi/180) * width_2 * 2
                elif angle_2 == 45:
                    len_x_2 = len_y_2 = math.cos(angle_2 * math.pi/180) * height_2 * 2
                else:
                    len_y_2 = math.cos(angle_2 * math.pi/180) * height_2 * 2
                    len_x_2 = math.cos(angle_2 * math.pi/180) * width_2 * 2


                top_y_2 = center_2[1] + abs(len_y_2 / 2)

                box_len_x.append(abs(len_x_2))
                box_len_y.append(abs(len_y_2))
                box_top_y.append(top_y_2)
                ellipse_width.append(width_2)
                ellipse_height.append(height_2)
                ellipse_angle.append(angle_2)
                # ellipse_height.append((m_x2,m_y2))
                frame_num.append(c)
                if width_2 == 0 or height_2 == 0:
                    bbox_r.append(0)
                else:
                    bbox_r.append(len_y_2 / len_x_2)
                # bbox_r.append((m_y2-new_m_y1)/(h_x2-h_x1))
                # print(bbox_r)
                # print(box_len_x)

                if flag_checkpoint == True:
                    # Check moving

                    if (b + 1) % video_frame == 0:
                        dis_label = b

                    box_center_dis = eucliDist(
                        (
                            box_len_x[count -
                                         video_frame * 2] / frame_width,
                            box_top_y[count -
                                         video_frame * 2] / frame_height,
                        ),
                        (
                            box_len_x[count] / frame_width,
                            box_top_y[count] / frame_height,
                        ),
                    )
                    # box_center_dis = eucliDist((box_len_x[c-video_frame] / frame_width, box_top_y[c-video_frame] / frame_height), (box_len_x[c-4-video_frame] / frame_width, box_top_y[c-4-video_frame] / frame_height))
                    # box_center_dis = eucliDist((box_len_x[dis_label] / frame_width, box_top_y[dis_label] / frame_height), (m_cX / frame_width, m_cY / frame_height))
                    # box_center_dis_prev = eucliDist((box_len_x[dis_label-1] / frame_width,box_top_y[dis_label-1]),(m_cX / frame_width,m_cY / frame_height))
                    # center_dis.append(box_center_dis)
                    # print("box_len_x:",box_len_x[dis_label])
                    # print("box_top_y:",box_top_y[dis_label])
                    # print(center_dis)

                    if (b + 1) % video_frame == 0:
                        last_5_box_r = (
                            bbox_r[count] +
                            bbox_r[count - 1] +
                            bbox_r[count - 2] +
                            bbox_r[count - 3] +
                            bbox_r[count - 4]
                        ) / 5
                        # last_5_center_dis = (center_dis[0] + center_dis[1] + center_dis[2] + center_dis[3] + center_dis[4])/5
                        last_5_center_dis = box_center_dis
                        front_5_frame_r = (
                            bbox_r[count - video_frame * 2] +
                            bbox_r[count - video_frame * 2 + 1] +
                            bbox_r[count - video_frame * 2 + 2] +
                            bbox_r[count - video_frame * 2 + 3] +
                            bbox_r[count - video_frame * 2 + 4]
                        ) / 5
                        data = np.array(
                            [front_5_frame_r, last_5_box_r, last_5_center_dis]
                        ).reshape(1, -1)
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
                        # np.savetxt( video_name + '.csv', np.c_[label],delimiter=',')
                        if flag2 == True:
                            frame = cv2.putText(
                                frame,
                                "Warning",
                                (int(frame_width / 20), int(frame_height / 5)),
                                cv2.FONT_HERSHEY_TRIPLEX,
                                0.5,
                                (0, 0, 255),
                            )
                        if test_mode:
                            frame = cv2.putText(
                                frame,
                                action,
                                (int(frame_width / 30), int(frame_height / 20)),
                                cv2.FONT_HERSHEY_TRIPLEX,
                                0.5,
                                (0, 0, 255),
                            )
                            cv2.imwrite(output_path + "%d.jpg" % (c), frame)
                    b += 1
                    print("b:", b)
                    # Check moving End

                    if b == video_frame:
                        flag_checkpoint = False
                        b = 0

                else:
                    if (
                        count >= video_frame * 2 and
                        2 >
                        abs(bbox_r[count - 4] - bbox_r[count]) >
                        checkpoint_threshold
                    ):
                        print("bbox_r[c]:", bbox_r[count])
                        print("bbox_r[c-4]:", bbox_r[count - 4])
                        flag_checkpoint = True
                        print(count)

            # Calculate End

            if flag2 == True:
                frame = cv2.putText(
                    frame,
                    "Warning",
                    (int(frame_width / 20), int(frame_height / 5)),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    (0, 0, 255),
                )
            if test_mode is False:
                if action is not None:
                    if action == "fall down":
                        frame = cv2.putText(
                            frame,
                            action,
                            (int(frame_width / 30), int(frame_height / 20)),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.5,
                            (0, 0, 250),
                        )
                        cv2.imwrite(output_path + "%d.jpg" % (c), frame)
                    else:
                        frame = cv2.putText(
                            frame,
                            action,
                            (int(frame_width / 30), int(frame_height / 20)),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.5,
                            (0, 180, 0),
                        )
                        cv2.imwrite(output_path + "%d.jpg" % (c), frame)
                else:
                    cv2.imwrite(output_path + "%d.jpg" % (c), frame)
                img_list.append(frame)

            # print("c:",c)
            c = c + 1
            end = time.time()
            # print("執行時間：%f 秒" % (end - start))
            # out.write(result)

            # 将影像保存到文件

            cv2.destroyAllWindows()
            # image.write_video(img_list, output_path + video_name + ".mp4", 12)
            np.savetxt(
                "../dataset/data/excel/ellipse/" + video_name + ".csv",
                np.c_[bbox_r, ellipse_width, ellipse_height, ellipse_angle, box_top_y, box_len_y, box_len_x,
                       frame_num],
                delimiter=",",
            )
        # print(c-count)
