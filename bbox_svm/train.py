#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import pycocotools

#声明类别，尽量保持
CLASS_NAMES =["__background__", "person",]
#  "bicycle",
#  "car",
#  "motorcycle",
#  "airplane",
#  "bus",
#  "train",
#  "truck",
#  "boat",
#  "traffic light",
#  "fire hydrant",
#  "stop sign",
#  "parking meter",
#  "bench",
#  "bird",
#  "cat",
#  "dog",
#  "horse",
#  "sheep",
#  "cow",
#  "elephant",
#  "bear",
#  "zebra",
#  "giraffe",
#  "backpack",
#  "umbrella",
#  "handbag",
#  "tie",
#  "suitcase",
#  "frisbee",
#  "skis",
#  "snowboard",
#  "sports ball",
#  "kite",
#  "baseball bat",
#  "baseball glove",
#  "skateboard",
#  "surfboard",
#  "tennis racket",
#  "bottle",
#  "wine glass",
#  "cup",
#  "fork",
#  "knife",
#  "spoon",
#  "bowl",
#  "banana",
#  "apple",
#  "sandwich",
#  "orange",
#  "broccoli",
#  "carrot",
#  "hot dog",
#  "pizza",
#  "donut",
#  "cake",
#  "chair",
#  "couch",
#  "potted plant",
#  "bed",
#  "dining table",
#  "toilet",
#  "tv",
#  "laptop",
#  "mouse",
#  "remote",
#  "keyboard",
#  "cell phone",
#  "microwave",
#  "oven",
#  "toaster",
#  "sink",
#  "refrigerator",
#  "book",
#  "clock",
#  "vase",
#  "scissors",
#  "teddy bear",
#  "hair drier",
#  "toothbrush"]
# 数据集路径

#tensorboard

DATASET_ROOT = '../dataset/data/8cam_data/train_mask_rcnn_1282_coco'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')

TRAIN_PATH = os.path.join(DATASET_ROOT, 'images/train2017')
VAL_PATH = os.path.join(DATASET_ROOT, 'images/val2017')

TRAIN_JSON = os.path.join(ANN_ROOT, 'instances_train2017.json')
VAL_JSON = os.path.join(ANN_ROOT, 'instances_val2017.json')
# 声明数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "final_train_coco_888": (TRAIN_PATH, TRAIN_JSON),
    "final_val_coco_888": (VAL_PATH, VAL_JSON),
    #"coco_my_train": (TRAIN_PATH, TRAIN_JSON),
    #"coco_my_val": (VAL_PATH, VAL_JSON),
}
#===========以下有两种注册数据集的方法，本人直接用的第二个plain_register_dataset的方式 也可以用register_dataset的形式==================
#注册数据集（这一步就是将自定义数据集注册进Detectron2）
def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   json_file=json_file,
                                   image_root=image_root)


#注册数据集实例，加载数据集中的对象实例
def register_dataset_instances(name, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco")


def load_dataset():
    return load_coco_json()

#=============================
# 注册数据集和元数据
def plain_register_dataset():
    #训练集
    DatasetCatalog.register("final_train_coco_888", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("final_train_coco_888").set(thing_classes=CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                    evaluator_type='coco', # 指定评估方式
                                                    json_file=TRAIN_JSON,
                                                    image_root=TRAIN_PATH)

    #DatasetCatalog.register("final_val_coco_888", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_2017_val"))
    #验证/测试集
    DatasetCatalog.register("final_val_coco_888", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("final_val_coco_888").set(thing_classes=CLASS_NAMES, # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                evaluator_type='coco', # 指定评估方式
                                                json_file=VAL_JSON,
                                                image_root=VAL_PATH)
# 查看数据集标注，可视化检查数据集标注是否正确，
#这个也可以自己写脚本判断，其实就是判断标注框是否超越图像边界
#可选择使用此方法
def checkout_dataset_annotation(name="final_val_coco_888"):
    #dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
    dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH)
    print(len(dataset_dicts))
    for i, d in enumerate(dataset_dicts,0):
        #print(d)
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
        vis = visualizer.draw_dataset_dict(d)
        #cv2.imshow('show', vis.get_image()[:, :, ::-1])
        cv2.imwrite('out/'+str(i) + '.jpg',vis.get_image()[:, :, ::-1])
        #cv2.waitKey(0)
        if i == 200:
            break


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    #args.config_file = "../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" 
    #args.config_file = "/home/ian/code/detectron2_repo/tools/output_2class/config.yaml" 
    args.config_file = "../states/detectron_model/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(args.config_file)   # 从config file 覆盖配置
    cfg.merge_from_list(args.opts)          # 从CLI参数 覆盖配置

    # 更改配置参数


    #cfg.MODEL.RESNETS.DEPTH = 34
    #cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
    
    #cfg.MODEL.ROI_HEADS.NAME = "CascadeROIHeads"

    
    cfg.DATASETS.TRAIN = ("final_train_coco_888",) # 训练数据集名称
    cfg.DATASETS.TEST = ("final_val_coco_888",)
    cfg.DATALOADER.NUM_WORKERS = 4  # 单线程

    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.MAX_SIZE_TRAIN = 720 # 训练图片输入的最大尺寸
    cfg.INPUT.MAX_SIZE_TEST = 720 # 测试数据输入的最大尺寸
    cfg.INPUT.MIN_SIZE_TRAIN = (150, 480) # 训练图片输入的最小尺寸，可以设定为多尺度训练
    cfg.INPUT.MIN_SIZE_TEST = 150
    #cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING，其存在两种配置，分别为 choice 与 range ：
    # range 让图像的短边从 512-768随机选择
    #choice ： 把输入图像转化为指定的，有限的几种图片大小进行训练，即短边只能为 512或者768
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'choice'
#  本句一定要看下注释！！！！！！！！
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.RETINANET.NUM_CLASSES = 2  # 类别数+1（因为有background，也就是你的 cate id 从 1 开始，如果您的数据集Json下标从 0 开始，这个改为您对应的类别就行，不用再加背景类！！！！！）
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256]]
    # cfg.MODEL.RETINANET.IN_FEATURES = ["p4", "p5", "p6","p7"]
    cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.05
    cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 2.0



    #cfg.MODEL.WEIGHTS="/home/yourstorePath/.pth"
    #cfg.MODEL.WEIGHTS = "../model/model_final_280758.pkl"    # 预训练模型权重  
    cfg.MODEL.WEIGHTS = "../states/detectron_model/model_final_mask_rcnn.pkl"
    #cfg.MODEL.WEIGHTS = "../model/model_final_5bd44e.pkl" 
    
    #cfg.MODEL.WEIGHTS = "/home/ian/code/detectron2_repo/model/model_final_mask_r50.pkl" 

    cfg.SOLVER.IMS_PER_BATCH = 20 # batch_size=2; iters_in_one_epoch = dataset_imgs/batch_size

    # 根据训练数据总数目以及batch_size，计算出每个epoch需要的迭代次数
    #9000为你的训练数据的总数目，可自定义
    ITERS_IN_ONE_EPOCH = int(9000 / cfg.SOLVER.IMS_PER_BATCH)
    #ITERS_IN_ONE_EPOCH = int(27000 / cfg.SOLVER.IMS_PER_BATCH)

    # 指定最大迭代次数
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 12) - 1 # 12 epochs，
    # 初始学习率
    cfg.SOLVER.BASE_LR = 0.002
    # 优化器动能
    cfg.SOLVER.MOMENTUM = 0.9
    #权重衰减
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    # 学习率衰减倍数
    cfg.SOLVER.GAMMA = 0.1
    # 迭代到指定次数，学习率进行衰减
    cfg.SOLVER.STEPS = (7000,)
    #cfg.SOLVER.STEPS = (21000,250000)
    # 在训练之前，会做一个热身运动，学习率慢慢增加初始学习率
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    # 热身迭代次数
    cfg.SOLVER.WARMUP_ITERS = 1000

    cfg.SOLVER.WARMUP_METHOD = "linear"
    # 保存模型文件的命名数据减1
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    # 迭代到指定次数，进行一次评估
    cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH 
    #cfg.TEST.EVAL_PERIOD = 100

    #cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg



def main(args):
    cfg = setup(args)
    plain_register_dataset()
    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )