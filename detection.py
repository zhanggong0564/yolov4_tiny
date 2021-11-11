import cv2
import numpy as np
import torch
from model.yolov4_tiny import Yolov4Tiny
from utils.utils_bbox import DecodeBox
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

        pass
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