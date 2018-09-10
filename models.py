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


class ResBlock(nn.Module):
    def __init__(self, in_C, out_C, repeat=1):
        super(__class__, self).__init__()
        self.module_list = nn.ModuleList([ConvLayer(in_C, out_C, stride=2, pool=False)])
        for i in range(repeat):
            self.module_list.append(ConvLayer(out_C, in_C, 1, pad=0, pool=False))
            self.module_list.append(ConvLayer(in_C, out_C, pool=False))

    def forward(self, x):
        x0 = self.module_list[0](x)
        for i in range(1, len(self.module_list), 2):
            x1 = self.module_list[i+1](self.module_list[i](x0))
            x0 += x1

        return x0


class Yolo3(nn.Module):
    def __init__(self, anchors=None, classes=80):
        super(__class__, self).__init__()

        predict_channel = 3 * (classes + 5)

        self.anchors = anchors
        self.bb_convlayer_1 = ConvLayer(3, 32, pool=False)   # bb: backbone
        self.bb_resblock_1 = ResBlock(32, 64, 1)
        self.bb_resblock_2 = ResBlock(64, 128, 2)
        self.bb_resblock_3 = ResBlock(128, 256, 8) # for dt3
        self.bb_resblock_4 = ResBlock(256, 512, 8) # for dt2
        self.bb_resblock_5 = ResBlock(512, 1024, 4)

        self.bb_convlayer_2 = ConvLayer(1024, 512, 1, pad=0, pool=False)
        self.bb_convlayer_3 = ConvLayer(512, 1024, pool=False)
        self.bb_convlayer_4 = ConvLayer(1024, 512, 1, pad=0, pool=False)
        self.bb_convlayer_5 = ConvLayer(512, 1024, pool=False)  
        self.bb_convlayer_6 = ConvLayer(1024, 512, 1, pad=0, pool=False) # branch to dt1 and dt2

        self.dt1_convlayer_1 = ConvLayer(512, 1024, pool=False)
        self.detector1 = ConvLayer(1024, predict_channel, 1, pad=0, bn=False, act=False, pool=False)

        self.bb_convlayer_7 = ConvLayer(512, 256, 1, pad=0, pool=False)  # followed by upsample and then concat 
        self.bb_convlayer_8 = ConvLayer(768, 256, 1, pad=0, pool=False)
        self.bb_convlayer_9 = ConvLayer(256, 512, pool=False)
        self.bb_convlayer_10 = ConvLayer(512, 256, 1, pad=0, pool=False)
        self.bb_convlayer_11 = ConvLayer(256, 512, pool=False) 
        self.bb_convlayer_12 = ConvLayer(512, 256, 1, pad=0, pool=False) # branch to dt2 and dt3
        
        self.dt2_convlayer_1 = ConvLayer(256, 512, pool=False)
        self.detector2 = ConvLayer(512, predict_channel, 1, pad=0, bn=False, act=False, pool=False)

        self.bb_convlayer_13 = ConvLayer(256, 128, 1, pad=0, pool=False)  # followed by upsample and then concat 
        self.bb_convlayer_14 = ConvLayer(384, 128, 1, pad=0, pool=False)
        self.bb_convlayer_15 = ConvLayer(128, 256, pool=False)
        self.bb_convlayer_16 = ConvLayer(256, 128, 1, pad=0, pool=False)
        self.bb_convlayer_17 = ConvLayer(128, 256, pool=False)
        self.bb_convlayer_18 = ConvLayer(256, 128, 1, pad=0, pool=False)

        self.dt3_convlayer_1 = ConvLayer(128, 256, pool=False)
        self.detector3 = ConvLayer(256, predict_channel, 1, pad=0, bn=False, act=False, pool=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.bb_convlayer_1(x)
        x_dt3 = self.bb_resblock_3(self.bb_resblock_2(self.bb_resblock_1(x)))

        x_dt2 = self.bb_resblock_4(x_dt3)

        x = self.bb_resblock_5(x_dt2)
        x = self.bb_convlayer_4(self.bb_convlayer_3(self.bb_convlayer_2(x)))
        x_dt123 = self.bb_convlayer_6(self.bb_convlayer_5(x))

        x = self.dt1_convlayer_1(x_dt123)
        y1 = self.detector1(x)

        x = self.bb_convlayer_7(x_dt123)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x_dt2], 1)
        x = self.bb_convlayer_10(self.bb_convlayer_9(self.bb_convlayer_8(x)))
        x_dt23 = self.bb_convlayer_12(self.bb_convlayer_11(x))

        x = self.dt2_convlayer_1(x_dt23)
        y2 = self.detector2(x)

        x = self.bb_convlayer_13(x_dt23)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x_dt3], 1)
        x = self.bb_convlayer_16(self.bb_convlayer_15(self.bb_convlayer_14(x)))
        x = self.bb_convlayer_18(self.bb_convlayer_17(x))

        x = self.dt3_convlayer_1(x)
        y3 = self.detector3(x)

        return y1, y2, y3


class Yolo3Tiny(nn.Module):
    def __init__(self, anchors=None):
        super(__class__, self).__init__()

        self.anchors = anchors
        self.bb_convlayer_1 = ConvLayer(3, 16)   # bb: backbone
        self.bb_convlayer_2 = ConvLayer(16, 32)
        self.bb_convlayer_3 = ConvLayer(32, 64)
        self.bb_convlayer_4 = ConvLayer(64, 128)  # for detector2
        self.bb_convlayer_5 = ConvLayer(128, 256, pool=False)
        self.bb_convlayer_6 = ConvLayer(256, 512, pool=True, pool_stride=1, pool_pad=0)
        self.bb_convlayer_7 = ConvLayer(512, 1024, pool=False)
        self.bb_convlayer_8 = ConvLayer(1024, 256, 1, pad=0, pool=False) # for detector1
        
        self.dt1_convlayer_1 = ConvLayer(256, 512, pool=False)
        self.detector1 =  ConvLayer(512, 255, 1, pad=0, bn=False, act=False, pool=False)

        self.dt2_convlayer_1 = ConvLayer(256, 128, 1, pad=0, pool=False)
        self.dt2_convlayer_2 = ConvLayer(384, 256, pool=False)
        self.detector2 =  ConvLayer(256, 255, 1, pad=0, bn=False, act=False, pool=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.bb_convlayer_1(x)
        x = self.bb_convlayer_2(x)
        x = self.bb_convlayer_3(x)
        x_dt2 = self.bb_convlayer_4(x)
        x_dt2 = self.bb_convlayer_5(x_dt2)
        x = F.max_pool2d(x_dt2, 2)
        x = self.bb_convlayer_6(x)
        x= F.interpolate(x, size=(x.size(2)+1, x.size(3)+1), mode='bilinear', align_corners=False)
        x = self.bb_convlayer_7(x)
        x_dt = self.bb_convlayer_8(x)

        x = self.dt1_convlayer_1(x_dt)
        y_dt1 = self.detector1(x)

        x = self.dt2_convlayer_1(x_dt)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x_dt2], 1)
        x = self.dt2_convlayer_2(x)
        y_dt2 = self.detector2(x)

        return y_dt1, y_dt2
        




