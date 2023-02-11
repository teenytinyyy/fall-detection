# coding=utf-8

# 加载一些基础包以及设置logger
from natsort import natsorted
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
import create_json as create_json
import files as files
import image as image


setup_logger()


if __name__ == '__main__':
    start = time.time()

    for num in range(24, 25):
        for num1 in range(1, 2):

            
            thresh_ratio = 0.2  # 20%閥值
            video_name = "data (" + str(num) + "_" + str(num1) + ")"
            video_name_new = str(num) + "_" + str(num1)
            # video_name = 'split' + str(num)
            # video_name_new = 'split' + str(num)
            #input_path = "../dataset/data/FDD_data_picture/" + video_name
            input_path = "../dataset/data/RGB_data/" + video_name
            output_path = '../dataset/data/json/mask_rcnn/mask_rcnn_1320/' + video_name_new + "/"
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            #images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in natsorted(
            #    glob.glob(input_path + "/*.jpg"))]
            # images = [cv2.imread(file) for file in natsorted(
            #     glob.glob(input_path + "/*.jpg"))]
            images = [cv2.imread("../dataset/data/test.jpg")]
            model_file_path = '../states/detectron_model/config.yaml'
            model_weights = "../states/detectron_model/model_final.pth"

            # model_file_path = '../states/detectron_model/mask_rcnn_1320/config.yaml'
            # model_weights = "../states/detectron_model/mask_rcnn_1320/model_final.pth"
            # model_file_path = '../states/detectron_model/mask_rcnn_R_50_FPN_3x.yaml'
            # model_weights = "../states/detectron_model/model_final_mask_rcnn.pkl"
            cfg = get_cfg()
            cfg.merge_from_file(model_file_path)
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
            # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7
            cfg.MODEL.WEIGHTS = model_weights
            predictor = DefaultPredictor(cfg)

            count = 0
            frame_num = 1
            bbox = []
            mask_box = []
            frame_nums = []
            check_point = False
            crop = False

            print("images#:", len(images))
            for idx, frame in enumerate(images):
                # if check_point == True:
                #     # print(mask_box[-1])
                #     prev_x1, prev_y1, prev_x2, prev_y2 = mask_box[-1]
                #     frame, crop_x1, crop_y1 = image.crop_area(frame, prev_x1, prev_y1, prev_x2, prev_y2, 100)
                #     crop = True
                #frame = cv2.equalizeHist(frame)
                #cv2.imwrite(output_path + "/" + video_name_new + "_" + "%d.jpg" % (frame_num), frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite("../dataset/data/test_gray.jpg", frame)
                exit()
                outputs = predictor(frame)
                # print(outputs)
                # Transfer relevant data to cpu
                single_prediction = outputs["instances"].to("cpu")._fields
                #print("single_prediction", single_prediction)
                person_selection_mask = single_prediction["pred_classes"] == 0
                predictions = {"pred_boxes": single_prediction["pred_boxes"].tensor[person_selection_mask].data.numpy(),
                                "scores": single_prediction["scores"][person_selection_mask].data.numpy(),
                                "pred_classes": single_prediction["pred_classes"][person_selection_mask].data.numpy(),
                                "pred_masks": single_prediction["pred_masks"][person_selection_mask].data.numpy()}

                if len(predictions["pred_classes"]) != 0:
                    if predictions["pred_classes"][0] == 0:
                        check_point = True
                        count += 1
                        box_x1, box_y1, box_x2, box_y2 = predictions["pred_boxes"][0]
                        mask_region = predictions["pred_masks"][0]

                        origin_img = np.full(frame.shape, 255).astype(np.uint8)
                        origin_img = Mask(mask_region).draw(
                            origin_img, color=(0, 0, 0), alpha=1)     
                        print(box_x1, box_y1, box_x2, box_y2)                 
                        #cv2.rectangle(origin_img, (int(box_x1), int(box_y1)), (int(box_x2), int(box_y2)), (0, 0, 255), 3)
                        cv2.imwrite("../dataset/data/test_mask.jpg", origin_img)

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
                        image_data = create_json.img_data(output_path + "/" + video_name_new + "_" + str(frame_num) + ".jpg")
                        json_file = "{}{}.json".format(output_path, picture_name)
                        create_json.create_json_file(Mask(mask_region).polygons().points, image_path, image_data, frame_height, frame_width, json_file)
                        print(mask_x1, mask_y1, mask_x2, mask_y2)
                        # cv2.rectangle(origin_img, (box_x1, box_y1),
                        #                 (box_x2, box_y2), (0, 255, 0), 2)

                        cv2.rectangle(origin_img, (mask_x1, mask_y1),
                                        (mask_x2, mask_y2), (0, 0, 255), 3)
                        cv2.imwrite("../dataset/data/test_mask_h.jpg", origin_img)

                        # cv2.rectangle(frame, (mask_x1, mask_y1),
                        #                 (mask_x2, mask_y2), (0, 0, 255), 3)
                        # frame = Mask(mask_region).draw(
                        #     frame, color=(0, 0, 255), alpha=0.5)
                        # bbox.append((box_x1, box_y1, box_x2, box_y2))
                        # cv2.imwrite(output_path + "/" + video_name_new + "_" + "%d_mask.jpg" % (frame_num), frame)

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
                # v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                # cv2.imwrite(output_path + "%d.jpg" % (c), out.get_image()[:, :, ::-1])

                # panoptic_seg, segments_info = predictor(frame)["panoptic_seg"]
                # print(panoptic_seg, segments_info)
                # v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                # out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
                # cv2.imwrite(output_path + "%d.jpg" % (c), out.get_image()[:, :, ::-1])
    end = time.time()
    print(end - start)
    print(count)
            # np.savetxt("../dataset/data/excel/mask_rcnn_1320/" + video_name_new +
            #            ".csv", np.c_[bbox, mask_box, frame_nums], delimiter=",")
    np.savetxt("../dataset/data/test_mask.csv", np.c_[horizontal_projection], delimiter=",")