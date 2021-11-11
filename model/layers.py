import torch.nn as nn
import torch




class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


'''
                    input
                      |
                  BasicConv
                      -----------------------
                      |                     |
                 route_group              route
                      |                     |
                  BasicConv                 |
                      |                     |
    -------------------                     |
    |                 |                     |
 route_1          BasicConv                 |
    |                 |                     |
    -----------------cat                    |
                      |                     |
        ----      BasicConv                 |
        |             |                     |
      feat           cat---------------------
                      |
                 MaxPooling2D
'''
class Resblock_body(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Resblock_body, self).__init__()
        self.conv1 = BasicConv(in_channels,out_channels,3)
        self.conv2 = BasicConv(out_channels//2,out_channels//2,3)
        self.conv3 =BasicConv(out_channels//2,out_channels//2,3)
        self.conv4 =BasicConv(out_channels,out_channels,1)
        self.maxpool = nn.MaxPool2d(2,2)
        self.out_channels =out_channels
        pass
    def forward(self,x):
        x = self.conv1(x)
        route = x

        x = torch.split(x,self.out_channels//2,dim=1)[1]
        x =self.conv2(x)
        route1 =x
        x = self.conv3(x)
        x = torch.cat([x,route1],dim=1)
        x = self.conv4(x)
        feat =x
        x = torch.cat([route,x],dim=1)
        x =self.maxpool(x)
        return x,feat


# conv+upsampleling
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.conv1 = BasicConv(in_channels, out_channels, 1)
        self.upsample_2x = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.upsample_2x(x)
        return x


class yolohead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_class):
        super(yolohead, self).__init__()
        self.conv1 = BasicConv(in_channels, mid_channels, 3)
        self.conv2 = nn.Conv2d(mid_channels, num_class, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class FPN(nn.Module):
    def __init__(self,num_class):
        super(FPN, self).__init__()
        self.conv1 = BasicConv(512,256,1)
        self.yolohead_p6 = yolohead(256,512,num_class)
        self.upsmaple = Upsample(256,128)

        self.yolohead_p4 = yolohead(384,256,num_class)

    def forward(self,inputs):
        x4,x6 = inputs
        x6 = self.conv1(x6)
        p6 = self.yolohead_p6(x6)

        p4 = self.upsmaple(x6)
        p4 = torch.cat([x4,p4],1)
        p4 = self.yolohead_p4(p4)
        return p6,p4





