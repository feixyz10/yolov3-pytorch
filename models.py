import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

class ConvLayer(nn.Module):
    def __init__(self, in_C, out_C, filter_size=3, stride=1, pad=1, bn=True, act=True, pool=True, pool_stride=2, pool_pad=0):
        super(__class__, self).__init__()
        self.bn = bn
        self.act = act
        self.pool = pool

        self.conv = nn.Conv2d(in_C, out_C, filter_size, stride=stride, padding=pad, bias=not self.bn)
        if self.bn: self.batchnorm = nn.BatchNorm2d(out_C)
        if self.act: self.activation = nn.LeakyReLU(0.1, inplace=True)
        if self.pool: self.maxpool = nn.MaxPool2d(2, pool_stride, pool_pad)

    def forward(self, x):
        x = self.conv(x)
        if self.bn: x = self.batchnorm(x)
        if self.act: x = self.activation(x)
        if self.pool: x = self.maxpool(x)
        
        return x


class Yolo3Tiny(nn.Module):
    def __init__(self, anchors=None):
        super(__class__, self).__init__()

        self.anchors = anchors
        self.bb_convlayer_1 = ConvLayer(3, 16)   # bb: backbone
        self.bb_convlayer_2 = ConvLayer(16, 32)
        self.bb_convlayer_3 = ConvLayer(32, 64)
        self.bb_convlayer_4 = ConvLayer(64, 128)  # for detector2
        self.bb_convlayer_5 = ConvLayer(128, 256)
        self.bb_convlayer_6 = ConvLayer(256, 512, pool=False)
        self.bb_convlayer_7 = ConvLayer(512, 1024, pool=False)
        self.bb_convlayer_8 = ConvLayer(1024, 256, 1, pad=0, pool=False) # for detector1
        
        self.dt1_convlayer_1 = ConvLayer(256, 512, pool=False)
        self.detector1 =  ConvLayer(512, 255, 1, pad=0, bn=False, act=False, pool=False)

        self.dt2_convlayer_1 = ConvLayer(256, 128, 1, pad=0, pool=False)
        self.dt2_convlayer_2 = ConvLayer(256, 256, pool=False)
        self.detector2 =  ConvLayer(256, 255, 1, pad=0, bn=False, act=False, pool=False)

    def forward(self, x):
        x = self.bb_convlayer_1(x)
        x = self.bb_convlayer_2(x)
        x = self.bb_convlayer_3(x)
        x_dt2 = self.bb_convlayer_4(x)
        x = self.bb_convlayer_5(x_dt2)
        x = self.bb_convlayer_6(x)
        x = self.bb_convlayer_7(x)
        x_dt = self.bb_convlayer_8(x)

        x = self.dt1_convlayer_1(x_dt)
        y_dt1 = self.detector1(x)

        x = self.dt2_convlayer_1(x_dt)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = torch.cat([x, x_dt2], 1)
        x = self.dt2_convlayer_2(x)
        y_dt2 = self.detector2(x)

        return y_dt1, y_dt2

        




