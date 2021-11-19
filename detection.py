import cv2
import numpy as np
import torch
from model.yolov4_tiny import Yolov4Tiny
from utils.utils_bbox import DecodeBox
import os
class Detction(object):
    def __init__(self,num_classes,weights_path,conf =0.5,nms_iou =0.5,model_input_shape=(416,416),use_gpu =True):
        super(Detction, self).__init__()
        self.__model_input_shape = model_input_shape
        self.num_classes  =num_classes
        self.__use_gpu =use_gpu
        self.__conf = conf
        self.__nms_iou =nms_iou
        self.__model = Yolov4Tiny(num_classes,3).cuda()

        self.__anchors =np.array([[ 10.,  14.],
                        [ 23.,  27.],
                        [ 37.,  58.],
                        [ 81.,  82.],
                        [135., 169.],
                        [344., 319.]])
        self.__anchors_mask = [[3, 4, 5], [1, 2, 3]]

        self.__decode = DecodeBox(self.__anchors,num_classes,self.__model_input_shape,self.__anchors_mask)
        try:
            self.__load_weights(weights_path)
        except:
            self.__load_weights_v1(weights_path)
        else:
            print('load model weights finsh')

        self.class_names = {0:'apple',1:'pear',2:'green apple',3:'orange'}
    @torch.no_grad()
    def detect(self,image):
        src_image_shape =image.shape[:2]
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,self.__model_input_shape)
        image = image/255.0#hwc
        image_data = np.expand_dims(np.transpose(image,(2,0,1)),0)
        images = torch.from_numpy(image_data)
        if self.__use_gpu:
            images = images.cuda().to(torch.float32)
            with torch.no_grad():
                outputs = self.__model(images)
                outputs = self.__decode.decode_box(outputs)
                results = self.__decode.non_max_suppression(torch.cat(outputs, 1),self.num_classes,self.__model_input_shape,src_image_shape,False,self.__conf,self.__nms_iou)
        results =results
        return results

    def __load_weights_v1(self,weights):
        state_dict = torch.load(weights)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        self.__model.load_state_dict(new_state_dict)
    def __load_weights(self,weights):
        state_dict = torch.load(weights)
        self.__model.load_state_dict(state_dict)

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        src_image_shape = image.shape[:2]
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image = cv2.resize(image,self.__model_input_shape)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(image,(2,0,1)),0)
        images = torch.from_numpy(image_data)
        if self.__use_gpu:
            images = images.cuda().to(torch.float32)
            with torch.no_grad():
                outputs = self.__model(images)
                outputs = self.__decode.decode_box(outputs)
                results = self.__decode.non_max_suppression(torch.cat(outputs, 1), self.num_classes,
                                                            self.__model_input_shape, src_image_shape, False,
                                                            self.__conf, self.__nms_iou)
        if results[0] is None:
            return

        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 4] * results[0][:, 5]
        top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return

    def __get_classes(self,classes_path):
        with open(classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names, len(class_names)