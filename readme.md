# yolov4_tiny

## 目录结构

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

## 网络结构

* Backbone:CSPdarknet
* Neck: SPP

## 训练所使用的tricks

- [x] Mixed Up

- [x] Mosaic

- [x] seblock

- [ ] Cutout:没有使用，会出现没有框的情况 不好计算loss

- [ ] CutMix

  激活函数

- [ ] Swith

- [ ] Mish

  

## 数据集

自己采集的水果数据集

总共2149张

train：1738

val: 194

test: 215

## 实验结果

VOC 计算方式的test 指标map结果。

|    模型     | 输入网络尺寸 |  mAP  |
| :---------: | :----------: | :---: |
| yolov4_tiny |   416x416    | 97.52 |
| yolov4_tiny |              |       |

COCO 计算方式的 test 指标map结果

|    模型     | 输入网络尺寸 |  AP   | AP50  | AP75  | AP_S | AP_M  | AP_L  |
| :---------: | :----------: | :---: | :---: | :---: | :--: | :---: | :---: |
| yolov4_tiny |   416x416    | 50.20 | 62.90 | 59.98 |  -1  | 36.73 | 58.94 |
| yolov4_tiny |              |       |       |       |      |       |       |

## 部署

 - [ ] opencv

 - [ ] opencv+cuda

 - [ ] tensorrt

  

######################################################################

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

水果数据集：

COCO eval: 

2021-11-12 17:08:46,272 - log.log - INFO - Epoch[020/50] Train loss:1.1952 Valid MAP0.50.5483LR:0.0009054634122156   
2021-11-12 17:08:46,553 - log.log - INFO - 11-12_17-08 done, best MAP0.5: 0.5576 in :17 epoch  
2021-11-12 17:09:01,748 - log.log - INFO - Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.2899844860501995   
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.5648710124835045   
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.22592612303669304   
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.0 
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.1991358695713297   
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.3678358611970964   
Average Recall       (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.28468115371148456   
Average Recall       (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.3863953081232493   
Average Recall       (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.3863953081232493   
Average Recall       (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.0   
Average Recall       (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.27308997844827587   
Average Recall       (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.476694647606383  



voc eval:

![image-20211117095952574](map_out/results/mAP.png)





加入各种数据增强实验如下：

加入tricks:

Mixed Up  :下降1个点的map,采用随机Mixed up 有涨点。0.几个点，修改数据增强的方式后涨了4个点到91->95->

+Mosaic 随机 96

+seblock 97.52





Label Smoothing:未实验

通过聚类的方式得到三个 anchor的宽高，但是map下降

############################################################





