import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from utils.dataaugment import get_transform
# from imp import reload
# reload(utils)
from utils.mosico import get_mosico_random_data
import random



class CustumDateset(Dataset):
    def __init__(self,annotation_lines_txt,input_shape,transform=None,mode='train',check=False):
        super(CustumDateset, self).__init__()
        self.annotation_lines_txt =annotation_lines_txt
        self.annotation_info = self.process_annotation_lines_txt()
        self.transform = transform
        self.input_shape = input_shape
        self.check =check
        self.use_mosico = True
        self.mode =mode
        pass

    def __len__(self):
        return len(self.annotation_info)

    def __getitem__(self, index):
        rand_mixup = random.choice([0, 1,2,3]) == 0
        if self.use_mosico and rand_mixup and self.mode=='train':
            mosico_index = np.random.choice(range(len(self.annotation_info)), 3).tolist()
            mosico_index.append(index)
            src_image, new_boxes = get_mosico_random_data(self.annotation_info,mosico_index,self.input_shape)
            h, w, c = src_image.shape
            image =src_image
            boxes = new_boxes
            if boxes.tolist()==[]:
                return 0,np.array([]),0,0,0
        else:
            line_infos = self.annotation_info[index].split()
            image_path = line_infos[0]
            boxes_info = line_infos[1:]
            boxes = np.array([list(map(int,box.split(','))) for box in boxes_info])
            src_image = cv2.imread(image_path)
            h,w,c = src_image.shape
            image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)

        image_id = torch.tensor([index])
        target_boxes = boxes[...,:4]
        target_labels = boxes[...,4]
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        targets = {
            'boxes':torch.tensor(target_boxes,dtype=torch.float32),
            'labels':torch.tensor(target_labels,dtype=torch.float32),
            'image_id':image_id,
            "iscrowd":iscrowd,
            "area":area
        }

        if self.transform:
            transformed = self.transform(image=image,bboxes=boxes[:,:4],class_labels=boxes[:,4])
            image = transformed['image']
            transformed_bboxes =np.array(transformed['bboxes'])
            transformed_class_labels =np.array(transformed['class_labels'])
            boxes = np.concatenate([transformed_bboxes,transformed_class_labels.reshape(len(transformed_bboxes),-1)],1)
            if self.check:
                for bbx in transformed_bboxes:
                    x1,y1 = int(bbx[0]),int(bbx[1])
                    x2,y2 = int(bbx[2]),int(bbx[3])
                    cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,255),1)
                cv2.imshow("image",image)
                cv2.waitKey()
                cv2.destroyAllWindows()
        image = image/255.0 #实验 去均值 方式he 直接、255
        image_tensor = ToTensorV2()(image=image)['image'].to(torch.float32)
        boxes = self.get_yolo_box(boxes)
        return  image_tensor,boxes,(h,w),src_image,targets
    def process_annotation_lines_txt(self):
        if os.path.exists(self.annotation_lines_txt):
            annotation_info = np.loadtxt(self.annotation_lines_txt,dtype=str,delimiter='\n').tolist()
            for i,an in enumerate(annotation_info):
                if len(an.split())<=1:
                    annotation_info.pop(i)
        else:
            raise("annotation path is not exist,please check path of annotation path")
        return annotation_info
    def get_yolo_box(self,boxs):
        boxs[:, [0, 2]] = boxs[:, [0, 2]] / self.input_shape[1]
        boxs[:, [1, 3]] = boxs[:, [1, 3]] / self.input_shape[0]

        boxs[:, 2:4] = boxs[:, 2:4] - boxs[:, 0:2]
        boxs[:, 0:2] = boxs[:, 0:2] + boxs[:, 2:4] / 2
        return boxs
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    targets =[]
    image_shape = (0,0)
    for img, box,image_shape,src_image,target in batch:
        if box.tolist()==[]:
            continue
        images.append(img)
        bboxes.append(box)
        targets.append(target)
    images = [i.unsqueeze(0) for i in images]
    images = torch.cat(images,0)
    return images, bboxes,image_shape,src_image,tuple(targets)


if __name__ == '__main__':
    Dts = CustumDateset("../datas/2007_train.txt",input_shape=(416,416),transform=get_transform())
    print(Dts[1])