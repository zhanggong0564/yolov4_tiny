# yolov4_tiny

.
├── cocoapi  
│   ├── coco_eval.py  
│   ├── coco_utils.py  
├── config  
│   ├── config.py  
├── datas   
│   ├── 2007_train.txt   
│   ├── 2007_val.txt   
│   ├── robot_classes.txt   
│   ├── voc_classes.txt   
│   └── yolov4_tiny_weights_coco.pth   
├── demo.py   
├── detection.py   
├── model   
│   ├── CSPdarknet53_tiny.py   
│   ├── layers.py  
│   └── yolov4_tiny.py   
├── readme.md  
├── scripts  
│   ├── get_map.py  
│   ├── kmeans_for_anchors.py  
│   └── voc_annotation.py  
├── train.py  
├── utils  
│   ├── coco_utils.py  
│   ├── datasets.py  
│   ├── __init__.py  
│   ├── model_trainer.py  
│   ├── utils_bbox.py  
│   ├── utils_map.py  
│   ├── utils.py  
│   └── yololoss.py  
└── VOCdevkit  
    └── VOC2007 -> /home/zhanggong/disk/Elements/data/robot/VOC2007    





目前已经完成以下工作：

1. 基本的dataset类，目前还没有做数据增强。
2. 搭建了yolov4_tiny模型，yoloss,decode等代码。
3. 添加cocoAPi 用来对验证集进行测试
4. 添加了log记录日志
5. 添加了断点预训练。

用了一个三分类的小数据进行测试：

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.6991  
Average Precision  (AP) @[ IoU=0.50      	| area=   all | maxDets=100 ] = 0.9727  
Average Precision  (AP) @[ IoU=0.75      	| area=   all | maxDets=100 ] = 0.9214  
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.0   
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.6990  
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.0   
Average Recall        (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.3410  
Average Recall        (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.7532  
Average Recall        (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.7532  
Average Recall        (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.0   
Average Recall        (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.7532  
Average Recall        (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.0    