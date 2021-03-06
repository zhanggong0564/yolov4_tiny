from utils.utils import *
from utils.datasets import CustumDateset,yolo_dataset_collate
from torch.utils.data import DataLoader
from model.yolov4_tiny import Yolov4Tiny
import torch
from utils.model_trainer import ModelTrainer
from utils.yololoss import YOLOLoss
from utils.utils import make_logger
from config.config import cfg
import argparse
set_seed(2021)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description="Training")
parser.add_argument('--lr',default=1e-3,type=float,help='learning rate')
parser.add_argument('--bs',default=16,type=int,help='training batchsize')
parser.add_argument('--resume',default=None,type=str,help='training resume path')
args = parser.parse_args()


if __name__ == '__main__':
    logger, log_dir = make_logger(cfg.exp_log,console_handler=False)
    #1 dataloder
    train_datasets = CustumDateset(cfg.train_dir,cfg.input_shape,transform=cfg.train_ts)
    valid_datasets = CustumDateset(cfg.val_dir,cfg.input_shape,transform=cfg.valid_ts,mode='val')

    train_loader =DataLoader(dataset=train_datasets,batch_size=args.bs,shuffle=True,num_workers=cfg.num_workers,collate_fn=yolo_dataset_collate)
    valid_loader = DataLoader(dataset=valid_datasets, batch_size=args.bs,shuffle=False,num_workers=cfg.num_workers,collate_fn=yolo_dataset_collate)

    #model
    model = Yolov4Tiny(cfg.num_classes,cfg.num_anchors)

    if cfg.pretrained and not args.resume:
        print("load pretrain model")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(cfg.pretrained)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    if cfg.use_multi_gpu:
        model = torch.nn.DataParallel(model)
    model.to(device)


    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    optimizer = torch.optim.SGD(model.parameters(),args.lr,momentum=0.9,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)

    loss_fn = YOLOLoss(cfg.anchors,cfg.num_classes,input_shape=cfg.input_shape,cuda=True,anchors_mask=cfg.anchors_mask,label_smoothing=cfg.label_smoothing)

    epoch_step = len(train_datasets) //args.bs
    epoch_step_val = len(valid_datasets) // args.bs

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['net'])  # ???????????????????????????
        optimizer.load_state_dict(checkpoint['optimizer'])  # ?????????????????????
        cfg.start_epoch = checkpoint['epoch']+1  # ???????????????epoch
        logger.info(f"start resume mode load from {cfg.start_epoch} epoch")

    # ????????????????????????????????????????????????????????????????????????cfg
    logger.info("cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n model:\n{}".format(
        cfg, loss_fn, scheduler, optimizer, model))
    MAP = []
    train_loss_utils = []
    valid_loss_utils = []
    for epoch in range(cfg.start_epoch,cfg.end_epoch):
        avg_loss = ModelTrainer.train(model,loss_fn,optimizer,epoch,cfg.end_epoch,epoch_step,train_loader,device,cfg)
        if epoch>=cfg.eval_epoch:
            cocoeval =ModelTrainer.valid_map(model,epoch,cfg.end_epoch,epoch_step_val,valid_loader,device,cfg)
            MAP05 = cocoeval.coco_eval['bbox'].stats[1]
            MAP.append(MAP05)
            logger.info(f"Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {cocoeval.coco_eval['bbox'].stats[0]} \n"
                        f"Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {cocoeval.coco_eval['bbox'].stats[1]} \n"
                        f"Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {cocoeval.coco_eval['bbox'].stats[2]} \n"
                        f"Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {cocoeval.coco_eval['bbox'].stats[3]} \n"
                        f"Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {cocoeval.coco_eval['bbox'].stats[4]} \n"
                        f"Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {cocoeval.coco_eval['bbox'].stats[5]} \n"
                        f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {cocoeval.coco_eval['bbox'].stats[6]} \n"
                        f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {cocoeval.coco_eval['bbox'].stats[7]} \n"
                        f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {cocoeval.coco_eval['bbox'].stats[8]} \n"
                        f"Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {cocoeval.coco_eval['bbox'].stats[9]} \n"
                        f"Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {cocoeval.coco_eval['bbox'].stats[10]} \n"
                        f"Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {cocoeval.coco_eval['bbox'].stats[11]} \n" )
            logger.info(f"Epoch[{epoch + 1:0>3}/{cfg.end_epoch}] "
                        f"Train loss:{avg_loss:.4f} Valid MAP0.5{MAP05:.4f}"
                        f"LR:{optimizer.param_groups[0]['lr']}")
            if cfg.best_MAP<MAP05:
                cfg.best_MAP=MAP05
                cfg.best_epoch = epoch
                torch.save(model.state_dict(),f'{cfg.weights_path}/best.pth')
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                "map":MAP05
            }
            torch.save(checkpoint,f'{cfg.weights_path}/last.pth')
            logger.info(
                f"{datetime.strftime(datetime.now(), '%m-%d_%H-%M')} done, best MAP0.5: {cfg.best_MAP:.4f} in :{cfg.best_epoch} epoch")
        else:
            valid_loss = ModelTrainer.valid_loss(model, loss_fn, epoch, cfg.end_epoch, epoch_step_val, valid_loader, device, cfg)
            valid_loss_utils.append(valid_loss)
            if cfg.best_loss>valid_loss:
                cfg.best_loss=valid_loss
                cfg.best_epoch = epoch
                torch.save(model.state_dict(),f'{cfg.weights_path}/best.pth')
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint,f'{cfg.weights_path}/last.pth')
            logger.info(
                f"{datetime.strftime(datetime.now(), '%m-%d_%H-%M')} done, best loss: {cfg.best_loss:.4f} in :{cfg.best_epoch} epoch")
        scheduler.step()
    plt.plot(range(0,cfg.eval_epoch),valid_loss_utils,label='loss')
    plt.plot(range(cfg.eval_epoch,cfg.end_epoch), MAP, label='map')
    plt.legend()
    plt.show()



