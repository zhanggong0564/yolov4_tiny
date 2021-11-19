from easydict import EasyDict
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.utils_bbox import DecodeBox

cfg = EasyDict()

cfg.train_dir = "./datas/2007_train.txt"                                                                #训练集info
cfg.val_dir = "./datas/2007_val.txt"                                                                    #c集info
cfg.pretrained = 'datas/yolov4_tiny_weights_coco.pth'
# cfg.pretrained = None                                                                                   #与训练权重的；路径
cfg.num_classes = 4

cfg.use_multi_gpu = True                                                                                #是否使用多GPU训练
cfg.input_shape = (416, 416)                                                                            #网络输入的shape
cfg.num_workers =8                                                                                      #多线程加载数据
cfg.start_epoch = 0                                                                                     #从第几个expoch开始
cfg.end_epoch = 200                                                                                     #总共训练多少个epoch
cfg.best_MAP = 0                                                                                        #设置保存模型的最小MAP
cfg.best_loss =10                                                                                       #设置保存模型的最小loss


cfg.eval_epoch = 5                                                                                      #设置验证map开始的epeoch

cfg.num_anchors = 3                                                                                     #anchors 的数量
cfg.best_epoch = 0

cfg.factor = 0.1
cfg.milestones = [30, 45]


cfg.anchors = np.array([[10., 14.],
                    [23., 27.],
                    [37., 58.],
                    [81., 82.],
                    [135., 169.],
                    [344., 319.]])                                                                       #anchors的宽高
cfg.anchors_mask = [[3, 4, 5], [1, 2, 3]]                                                                #对应的anchors的mask

cfg.decode = DecodeBox(cfg.anchors,
                       cfg.num_classes,
                       cfg.input_shape,
                       cfg.anchors_mask)                                                                  #解码的类

cfg.exp_log = "../logs/exp"                                                                               #记录日志的路径
cfg.weights_path = "../logs"                                                                              #权重的路径

cfg.train_ts = A.Compose([
    A.Resize(height=416,width=416),
    A.HorizontalFlip(p=0.5),
    A.RandomGamma(),
    A.VerticalFlip(p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(p=0.5),
        # A.ColorJitter(brightness=0.07, contrast=0.07,
        #               saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        A.Cutout(),
        A.CLAHE(),
        A.Blur()
    ]),
    A.RandomRotate90(p=0.5),
    # A.RandomBrightnessContrast(p=0.5),

    ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels']))                         #训练集的增强方式
cfg.valid_ts = A.Compose([
        A.Resize(height=416,width=416),
    ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels']))                         #验证集的增强方式
###############################data augument
cfg.use_mixup =True
cfg.label_smoothing = 0.05