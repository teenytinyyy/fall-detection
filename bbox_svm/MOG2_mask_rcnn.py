from MOG2 import ImageProcessorCloseLoop
import cv2
from natsort import natsorted
from imantics import Mask
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import os
import time
import numpy as np
from detectron2.utils.logger import setup_logger
import csv
import math
import glob
import create_json as create_json
import files as files
import image as image

setup_logger()


if __name__ == "__main__":
    for NUM in range(22, 24):
        for NUM1 in range(1, 9):
            video_name_new = str(NUM) + "_" + str(NUM1)
            cap = cv2.VideoCapture(r'../dataset/data/8cam_dataset/chute{}/cam{}_Trim.avi'.format(NUM, NUM1))
            output_path = '../dataset/data/MOG2_mask_rcnn/gray/mask_rcnn_1320/' + video_name_new + "/" + video_name_new + "_"
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            proc = ImageProcessorCloseLoop(tracker_type="MIL")

            model_file_path = '../states/detectron_model/mask_rcnn_1320/config.yaml'
            model_weights = "../states/detectron_model/mask_rcnn_1320/model_final.pth"
            #model_file_path = '../states/detectron_model/mask_rcnn_R_50_FPN_3x.yaml'
            #model_weights = "../states/detectron_model/model_final_mask_rcnn.pkl"
            cfg = get_cfg()
            cfg.merge_from_file(model_file_path)
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
            cfg.MODEL.WEIGHTS = model_weights
            predictor = DefaultPredictor(cfg)

            box_count = 0
            thresh_ratio = 0.1
            frame_num = 0
            detec_bbox = []
            mask_box = []
            frame_nums = []
            check_point = False
            crop = False

            while True:
                max_area = 0
                MOG2_checkpoint = False
                ret, frame = cap.read()
                if ret == False:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                bboxes = proc.process_one_frame(frame, frame_num, output_path)
                for bbox in bboxes:
                    x, y, w, h = bbox
                    area = abs(w) * abs(h)
                    if area > max_area:
                        max_area = area
                        x1, y1, w1, h1 = x, y, w, h
                        MOG_bbox = x1, y1, x1 + w1, y1 + h1
                cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 255, 255), 2)
                #cv2.imwrite("../dataset/data/MOG2_mask_rcnn/original/" + video_name_new + "_%d.jpg" % (frame_num), frame)
                #cv2.imwrite(output_path + str(frame_num) + "_MOGbox.jpg", frame)

                if max_area >= 2000:
                    MOG2_checkpoint = True
                #print(MOG2_checkpoint)
                # use detectron2
                if MOG2_checkpoint == True:
                    #print(frame_num, max_area)
                    outputs = predictor(frame)
                    single_prediction = outputs["instances"].to("cpu")._fields
                    person_selection_mask = single_prediction["pred_classes"] == 1
                    predictions = {"pred_boxes": single_prediction["pred_boxes"].tensor[person_selection_mask].data.numpy(),
                                    "scores": single_prediction["scores"][person_selection_mask].data.numpy(),
                                    "pred_classes": single_prediction["pred_classes"][person_selection_mask].data.numpy(),
                                    "pred_masks": single_prediction["pred_masks"][person_selection_mask].data.numpy()}
                    # cv2.imwrite(output_path + str(frame_num) + ".jpg", frame)

                    
                    # get bbox
                    nearest_box_idx = 0
                    Max_IOU = 0
                    # print(len(predictions["pred_boxes"]), predictions["pred_boxes"])
                    for i in range(len(predictions["pred_boxes"])):
                        IOU = image.calculate_IOU(predictions["pred_boxes"][i], MOG_bbox)
                        #print(frame_num, IOU, nearest_box_idx, predictions["pred_boxes"][i], MOG_bbox)
                        if IOU > Max_IOU:
                            Max_IOU = IOU
                            nearest_box_idx = i
                        #print(frame_num, Max_IOU, nearest_box_idx, predictions["pred_boxes"][i], MOG_bbox)
                    if Max_IOU != 0:
                    # if len(predictions["pred_classes"]) != 0:
                        if predictions["pred_classes"][nearest_box_idx] == 1:
                            # print(frame_num, nearest_box_idx)
                            cv2.imwrite(output_path + str(frame_num) + ".jpg", frame)
                            check_point = True
                            box_count += 1
                            box_x1, box_y1, box_x2, box_y2 = predictions["pred_boxes"][nearest_box_idx]
                            # if len(predictions["pred_boxes"]) > 1:
                            #     box1, box2, box3, box4 = predictions["pred_boxes"][0]
                            #     cv2.rectangle(frame, (box1, box2),
                            #                 (box3, box4), (0, 255, 255), 3)
                            mask_region = predictions["pred_masks"][nearest_box_idx]
                            # 垂直投影修正mask    
                            horizontal_projection = np.sum(mask_region, axis=0)
                            y_max = np.max(horizontal_projection)
                            histogram_thresh = y_max * thresh_ratio
                            for i in range(len(horizontal_projection)):
                                if horizontal_projection[i] >= histogram_thresh:
                                    for j in range(len(mask_region)):
                                        mask_region[j][0: i] = False
                                    break

                            for i in range(len(horizontal_projection)-1, 0, -1):
                                if horizontal_projection[i] >= histogram_thresh:
                                    for j in range(len(mask_region)):
                                        mask_region[j][i: -1] = False
                                    break

                            mask_x1, mask_y1, mask_x2, mask_y2 = Mask(mask_region).polygons().bbox()
                            frame_width = int(frame.shape[1])
                            frame_height = int(frame.shape[0])
                            picture_name = video_name_new + "_" + str(frame_num)
                            image_path = picture_name +".jpg"
                            # image_data = create_json.img_data(input_path + "/" + str(frame_num) + ".jpg")
                            image_data = create_json.img_data(output_path + str(frame_num) + ".jpg")
                            json_file = "{}{}.json".format(output_path, str(frame_num))
                            create_json.create_json_file(Mask(mask_region).polygons().points, image_path, image_data, frame_height, frame_width, json_file)

                            # cv2.rectangle(frame, (box_x1, box_y1),
                            #               (box_x2, box_y2), (0, 255, 0), 2)
                            cv2.rectangle(frame, (mask_x1, mask_y1),
                                            (mask_x2, mask_y2), (0, 0, 255), 3)
                            frame = Mask(mask_region).draw(
                                frame, color=(0, 0, 255), alpha=0.5)
                            detec_bbox.append((box_x1, box_y1, box_x2, box_y2))
                            cv2.imwrite(output_path + str(frame_num) + "_mask.jpg", frame)

                            if crop == True:
                                mask_box.append((mask_x1 + crop_x1, mask_y1 + crop_y1, mask_x2 + crop_x1, mask_y2 + crop_y1))
                            else:
                                mask_box.append((mask_x1, mask_y1, mask_x2, mask_y2))
                            frame_nums.append(frame_num)
                            # print(frame_num)
                        else:
                            check_point = False
                    else:
                        check_point = False

                    # cv2.imwrite(output_path + "/" + video_name_new + "_" + "%d_mask.jpg" % (frame_num), frame)
                    crop = False
                frame_num += 1
            # end = time.time()
            # print(end - start)
            # print(count)

            np.savetxt("../dataset/data/excel/MOG2_mask_rcnn/gray/mask_rcnn_1320/" + video_name_new + ".csv", np.c_[detec_bbox, mask_box, frame_nums], delimiter=",")
                


