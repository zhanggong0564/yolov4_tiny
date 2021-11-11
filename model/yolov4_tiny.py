from model.layers import *
from model.CSPdarknet53_tiny import darknet53_tiny

class Yolov4Tiny(nn.Module):
    def __init__(self,num_classes,anchors_nums):
        super(Yolov4Tiny, self).__init__()
        self.backbone  = darknet53_tiny(None)
        self.fpn = FPN(anchors_nums*(5+num_classes))
    def forward(self,x):
        outputs = self.backbone(x)
        p6,p4 = self.fpn(outputs)
        return p6,p4
if __name__ == '__main__':
    model = Yolov4Tiny(4)
    x = torch.randn(4,3,416,416)
    p6,p4 = model(x)
    print(p6.shape)
    print(p4.shape)