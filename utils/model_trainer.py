import torch
import numpy as np
from collections import Counter
from utils.yololoss import YOLOLoss
from tqdm import tqdm
from utils.utils import get_lr
from cocoapi.coco_utils import get_coco_api_from_dataset
from cocoapi.coco_eval import CocoEvaluator
from utils.utils import evaluate
from utils.mixup import mix_up
import random

class ModelTrainer(object):
    @staticmethod
    def train(model, yolo_loss, optimizer, epoch, Epoch,epoch_step, dataloader, device,cfg):
        '''
        :param model: yolomodel
        :param yolo_loss:
        :param optimizer:
        :param epoch: 当前进行到第几个epoch
        :param epoch_step: 一个epoch需要迭代多少次 total_num/bachsize
        :param dataloader:
        :param Epoch: 总训练论数
        :param cuda: 是否使用cuda
        :return:
        '''
        loss = 0
        model.train()
        with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(dataloader):
                if iteration >= epoch_step:
                    break

                images, targets = batch[0], batch[1]
                images = images.to(device)
                targets = [torch.from_numpy(ann).to(device).to(torch.float32) for ann in targets]
                rand_mixup = random.choice([0, 1, 2]) == 0 #设置随机mixup
                if cfg.use_mixup and rand_mixup:
                    images,label_a,label_b,lam =mix_up(images,targets)
                # ----------------------#
                #   清零梯度
                # ----------------------#
                optimizer.zero_grad()
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs = model(images)

                loss_value_all = 0
                num_pos_all = 0
                # ----------------------#
                #   计算损失
                # ----------------------#
                if cfg.use_mixup and rand_mixup:
                    for l in range(len(outputs)):
                        loss_item_a, num_pos = yolo_loss(l, outputs[l], label_a)
                        loss_item_b, num_pos = yolo_loss(l, outputs[l], label_b)
                        loss_item = lam*loss_item_a+(1-lam)*loss_item_b
                        loss_value_all += loss_item
                        num_pos_all += num_pos

                for l in range(len(outputs)):
                    loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                    loss_value_all += loss_item
                    num_pos_all += num_pos
                loss_value = loss_value_all / num_pos_all

                # ----------------------#
                #   反向传播
                # ----------------------#
                loss_value.backward()
                optimizer.step()

                loss += loss_value.item()

                pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)
        return loss / epoch_step

    @staticmethod
    @torch.no_grad()
    def valid_map(model, epoch,Epoch, epoch_step_val,dataloader, device,cfg):
        coco = get_coco_api_from_dataset(dataloader.dataset)
        iou_types = ["bbox"]  # iou_types = ["bbox"]
        coco_evaluator = CocoEvaluator(coco, iou_types)
        with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(dataloader):
                if iteration >= epoch_step_val:
                    break
                with torch.no_grad():
                    coco_resluts = evaluate(batch, device, model, cfg.decode,cfg.num_classes,(416,416))
                    coco_evaluator.update(coco_resluts)
                pbar.set_postfix_str("start coco_evaluator map")
                pbar.update(1)
            coco_evaluator.synchronize_between_processes()
            # accumulate predictions from all images
            coco_evaluator.accumulate()
            coco_evaluator.summarize()

        return coco_evaluator
    @staticmethod
    @torch.no_grad()
    def valid_loss(model,yolo_loss,epoch,Epoch, epoch_step_val,dataloader, device,cfg):
        loss = 0
        model.eval()
        with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(dataloader):
                if iteration >= epoch_step_val:
                    break
                with torch.no_grad():
                    images, targets = batch[0], batch[1]
                    images = images.to(device)
                    targets = [torch.from_numpy(ann).to(device).to(torch.float32) for ann in targets]
                    # ----------------------#
                    #   前向传播
                    # ----------------------#
                    outputs = model(images)
                loss_value_all = 0
                num_pos_all = 0
                # ----------------------#
                #   计算损失
                # ----------------------#
                for l in range(len(outputs)):
                    loss_item, num_pos = yolo_loss(l, outputs[l], targets)
                    loss_value_all += loss_item
                    num_pos_all += num_pos
                loss_value = loss_value_all / num_pos_all
                loss += loss_value.item()

                pbar.set_postfix(**{'loss': loss / (iteration + 1)})
                pbar.update(1)
        return loss / epoch_step_val