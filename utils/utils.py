import numpy as np
import torch
import random
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt

def set_seed(seed = 2021):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark =True

class Logger(object):
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)#log.log
        self.log_name = log_name if log_name else "root"
        self.out_path = path_log

        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self,console=True):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # 配置文件Handler
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        if console:
            # 配置屏幕Handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            # console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(console_handler)


        # 添加handler
        logger.addHandler(file_handler)

        return logger
def check_data_dir(path):
    assert os.path.exists(path),"\n\n路径不存在，当前变量中指定的路径是：\n{}\n请检查相对路径的设置，或者文件是否存在".format(os.path.abspath(path))


def make_logger(out_dir,console_handler=True):
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(out_dir, time_str)  # 根据config中的创建时间作为文件夹名
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    path_log = os.path.join(log_dir,"log.log")
    logger  = Logger(path_log)
    logger = logger.init_logger(console_handler)
    return logger,log_dir
def plot_line(x,t_y,v_y,mode,out_dir):
    plt.plot(x, t_y, label='Train')
    plt.plot(x, v_y, label='Valid')
    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)
    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

@torch.no_grad()
def evaluate(inputs,device,model,decode,num_classes,model_input_shape):
    images, src_image_shape = inputs[0], inputs[2]
    images = images.to(device).to(torch.float32)
    with torch.no_grad():
        outputs = model(images)
        outputs = decode.decode_box(outputs)#(y1,x1,y1,y2)
        results = decode.non_max_suppression(torch.cat(outputs, 1),num_classes,model_input_shape,src_image_shape,False,0.5,0.5)
    # results = {'boxes':0,'classes':0,'conf':0}
    res ={}
    for i in range(len(results)):
        if results[i]==[]:
            boxes = torch.tensor([]).resize(0,4)
            labels =torch.tensor([]).resize(0,4)
            scores = torch.tensor([]).resize(0,4)
        else:
            bb = torch.tensor(results[i][...,:4],requires_grad=False)
            boxes = torch.full_like(bb,0)
            boxes[...,0] = bb[...,1]
            boxes[..., 1] = bb[..., 0]
            boxes[..., 2] = bb[..., 3]
            boxes[..., 3] = bb[..., 2]
            labels = torch.tensor(results[i][...,6],requires_grad=False)
            scores = torch.tensor(results[i][...,4]*results[i][...,5],requires_grad=False)
        coco_resluts = {
            'boxes':boxes,
            'labels':labels,
            'scores':scores
        }
        res[inputs[-1][i]["image_id"].item()] = coco_resluts
    return res


