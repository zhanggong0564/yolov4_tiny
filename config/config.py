from easydict import EasyDict
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.utils_bbox import DecodeBox

cfg = EasyDict()

cfg.train_dir = "./datas/2007_train.txt"
cfg.val_dir = "./datas/2007_val.txt"
# cfg.pretrained = 'datas/yolov4_tiny_weights_coco.pth'
cfg.pretrained = None
cfg.num_classes = 4

cfg.use_multi_gpu = True
cfg.input_shape = (416, 416)
cfg.num_workers =4
cfg.start_epoch = 0
cfg.end_epoch = 100
cfg.best_MAP = 0
cfg.best_loss =10


cfg.eval_epoch = 50

cfg.num_anchors = 3
cfg.best_epoch = 0
cfg.decode = DecodeBox(np.array([[10., 14.],
                                [23., 27.],
                                [37., 58.],
                                [81., 82.],
                                [135., 169.],
                                [344., 319.]]),
                       3,
                       (416,416),
                       [[3, 4, 5], [1, 2, 3]])

cfg.anchors = np.array([[10., 14.],
                    [23., 27.],
                    [37., 58.],
                    [81., 82.],
                    [135., 169.],
                    [344., 319.]])
cfg.anchors_mask = [[3, 4, 5], [1, 2, 3]]
cfg.exp_log = "../logs/exp"
cfg.weights_path = "../logs"

cfg.train_ts = A.Compose([
        A.Resize(height=416,width=416),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels']))
cfg.valid_ts = A.Compose([
        A.Resize(height=416,width=416),
    ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels']))
