from model.layers import *
from model.CSPdarknet53_tiny import darknet53_tiny

class Yolov4Tiny(nn.Module):
    def __init__(self,num_classes,anchors_nums,deploy=False):
        super(Yolov4Tiny, self).__init__()
        self.backbone  = darknet53_tiny(None)
        self.fpn = FPN(anchors_nums*(5+num_classes))
        self.seblock_x4 = se_block(256)
        self.seblock_x6 = se_block(512)
        self.deploy =deploy
    def forward(self,x):
        outputs = self.backbone(x)#26,13 x4 x6
        ##seblock
        x4, x6 = outputs
        x4 = self.seblock_x4(x4)
        x6 = self.seblock_x6(x6)
        outputs = x4,x6

        p6,p4 = self.fpn(outputs)#13,26
        if self.deploy:
            out0 = p6.view(1, 3, 9, 13, 13).permute(0, 1, 3, 4, 2).contiguous()
            out0 = out0.view(-1,self.num_classes+5)
            out1 = p4.view(1, 3, 9, 26, 26).permute(0, 1, 3, 4, 2).contiguous()
            out1 = out1.view(-1, self.num_classes + 5)
            out = torch.cat((out0,out1),0)
            return out
        return p6,p4
if __name__ == '__main__':
    model = Yolov4Tiny(4)
    x = torch.randn(4,3,416,416)
    p6,p4 = model(x)
    print(p6.shape)
    print(p4.shape)