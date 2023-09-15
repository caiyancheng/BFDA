from torch import nn
import torch

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models

import pdb
def conv3x3(in_planes, out_planes, pad, dilation, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=pad, dilation=dilation, bias=True)
def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)
class DDCPP(nn.Module):
    def __init__(self, input_channel):
        super(DDCPP, self).__init__()
        self.reduced_conv = conv3x3(input_channel, 256, 1, 1)
        self.reduced_bn = nn.BatchNorm2d(256)
        self.ddc_x1 = conv3x3(256, 256, 2, 2)
        self.ddc_bn1 = nn.BatchNorm2d(256)
        self.ddc_x2 = conv3x3(512, 256, 4, 4)
        self.ddc_bn2 = nn.BatchNorm2d(256)
        self.ddc_x3 = conv3x3(768, 256, 8, 8)
        self.ddc_bn3 = nn.BatchNorm2d(256)
        self.post_conv = conv1x1(1024, 512)
        self.post_bn = nn.BatchNorm2d(512)
        self.pool1_conv = conv1x1(512, 128)
        self.pool1_bn = nn.BatchNorm2d(128)
        self.pool2_conv = conv1x1(512, 128)
        self.pool2_bn = nn.BatchNorm2d(128)
        self.pool3_conv = conv1x1(512, 128)
        self.pool3_bn = nn.BatchNorm2d(128)
        self.pool4_conv = conv1x1(512, 128)
        self.pool4_bn = nn.BatchNorm2d(128)
        self.conv2 = conv1x1(512, 512)
        self.bn2 = nn.BatchNorm2d(512)

        self.conv_cls = conv1x1(512, 2)
        #self.fc = nn.Linear(128, 2)

    def forward(self, x):
        # reduced_x
        x_r = F.relu(self.reduced_bn(self.reduced_conv(x)))
        #ddc x1
        x1 = F.relu(self.ddc_bn1(self.ddc_x1(x_r)))
        x1_c = torch.cat((x_r, x1), 1)
        # ddc x2
        x2 = F.relu(self.ddc_bn2(self.ddc_x2(x1_c)))
        x2_c = torch.cat((x1_c, x2), 1)
        # ddc x3
        x3 = F.relu(self.ddc_bn3(self.ddc_x3(x2_c)))

        #all concat
        x1_p = torch.cat((x_r, x1), 1)
        x2_p = torch.cat((x1_p, x2), 1)
        x3_p = torch.cat((x2_p, x3), 1)
        #post layers
        x_post = F.relu(self.post_bn(self.post_conv(x3_p)))

        #domain classifier
        x_p = F.relu(self.bn2(self.conv2(x_post)))
        #x = F.avg_pool2d(x_p, (x_p.size(2), x_p.size(3)))
        #x = x.view(-1, 128)
        x = self.conv_cls(x_p)

        return x

class ConvBNReLU(nn.Module):#CBR块

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):#256,256
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                # out_chan//4,
                out_chan//2,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//2,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)#torch.Size([14, 256, 128, 256])
        feat = self.convblk(fcat)#torch.Size([14, 256, 128, 256])
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class fc_discriminator(nn.Module):
    def __init__(self, num_classes, ndf=128, ):
        super(fc_discriminator, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
        )#64倍下采样

        self.branch2 = nn.Sequential(
            nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),#new
            nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1, dilation=1),
        )#8倍下采样

        self.branch3 = nn.Sequential(
        nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1, padding=1, dilation=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1, dilation=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),#new
        nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1, dilation=1),
    )#4倍下采样

        self.branch4 = nn.Sequential(
            nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),  # new
            nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1, dilation=1),
        )  # 2倍下采样

        self.upsample = nn.Upsample(size=4)
        self.downsample = nn.MaxPool2d(kernel_size=5, stride=4, padding=2)
        # self.downsample = nn.down
        self.ffm = FeatureFusionModule(2, 2)
        self.final = nn.Conv2d(2,1,kernel_size=3,stride=1,padding=1)
        self.init_weight()

    def forward(self, x):#[1,256,256,256]
        #x=torch.cat([self.upsample(self.branch1(x)), self.downsample(self.branch2(x))], dim=1) #[1,1,8,8] [1,1,32,32]
        x = self.branch4(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

#--------Deeplab-v2--------------
def get_fc_discriminator_add(num_classes, ndf=128):#32倍下采样
    return nn.Sequential(
        nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
    )

# def get_fc_discriminator_add(num_classes, ndf=128):
#     return nn.Sequential(
#         nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1),
#     )

#--------UNET--------------D1--------------------
# def get_fc_discriminator(num_classes, ndf=128):
#     return nn.Sequential(
#         nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=2, dilation=2),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=2, dilation=2),
#     )

#--------UNET--------------D2--------------------
def get_fc_discriminator(num_classes, ndf=128):#32倍下采样
    return nn.Sequential(
        nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        # nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1, padding=1, dilation=1),
        # nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        # nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1, dilation=1),
        # nn.LeakyReLU(negative_slope=0.2, inplace=True),#new
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),  # new
        nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1, dilation=1),
    )

# --------UNET--------------D3--------------------
# def get_fc_discriminator(num_classes, ndf=128):#4倍下采样
#     return nn.Sequential(
#         nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=1, padding=1, dilation=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1, dilation=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),#new
#         nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1, dilation=1),
#     )


#--------UNET--------------D4--------------------
# def get_fc_discriminator(num_classes, ndf=128):
#     return nn.Sequential(
#         nn.Conv2d(num_classes, ndf, kernel_size=3, stride=1, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=2, dilation=2),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=2, dilation=2),
#     )

# def get_fc_discriminator(num_classes, ndf=128):
#     return nn.Sequential(
#         nn.Conv2d(num_classes, ndf, kernel_size=3, stride=1, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=2, dilation=2),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=2, dilation=2),
#     )

if __name__ == "__main__":
    a=torch.randn([12,18,256,256])
    dis_model = fc_discriminator(num_classes=18)
    x=dis_model(a)