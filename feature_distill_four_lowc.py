# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
class F_D(nn.Module):
    def __init__(self, input_channel, output_channel=3):
        super(F_D, self).__init__()
        self.reduced_conv = conv3x3(input_channel, 18, 1, 1, 2)
        self.reduced_bn = nn.BatchNorm2d(18)

        self.ddc_x1 = conv3x3(36, 18, 2, 2, 2)
        self.ddc_bn1 = nn.BatchNorm2d(18)

        self.ddc_x2 = conv3x3(36, 18, 4, 4, 2)
        self.ddc_bn2 = nn.BatchNorm2d(18)

        self.ddc_x3 = conv3x3(36, 18, 4, 4)
        self.ddc_bn3 = nn.BatchNorm2d(18)

        self.post_conv = conv1x1(90, 32)
        self.post_bn = nn.BatchNorm2d(32)

        self.pool1_conv = conv1x1(32, 32)
        self.pool1_bn = nn.BatchNorm2d(32)

        self.pool2_conv = conv1x1(32, 32)
        self.pool2_bn = nn.BatchNorm2d(32)

        # self.pool3_conv = conv1x1(82, 32)
        # self.pool3_bn = nn.BatchNorm2d(32)

        # self.pool4_conv = conv1x1(82, 32)
        # self.pool4_bn = nn.BatchNorm2d(32)

        self.conv2 = conv1x1(96, 32)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv_cls = conv1x1(32, output_channel)
        #self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x_first, x_second, x_third, x_forth = x
        # reduced_x
        x0 = F.interpolate(x_first, size=(256, 256), mode='bilinear')
        x_second = F.interpolate(x_second, size=(128, 128), mode='bilinear')
        x_third = F.interpolate(x_third, size=(64, 64), mode='bilinear')
        x_forth = F.interpolate(x_forth, size=(32, 32), mode='bilinear')
        x1 = F.relu(self.reduced_bn(self.reduced_conv(x0)))#[4,32,256,256]
        x_f_1 = torch.cat((x1,x_second),1)
        #ddc x1
        x2 = F.relu(self.ddc_bn1(self.ddc_x1(x_f_1)))
        x_f_2 = torch.cat((x2, x_third), 1)#[4,64,256,256]
        # ddc x2
        x3 = F.relu(self.ddc_bn2(self.ddc_x2(x_f_2)))
        x_f_3 = torch.cat((x3, x_forth), 1)#[4,96,256,256]
        # # ddc x3
        x4 = F.relu(self.ddc_bn3(self.ddc_x3(x_f_3)))
        #all concat
        x1 = F.interpolate(x1, size=(x0.size(2), x0.size(3)), mode='bilinear') #[4,64,224,224]
        x2 = F.interpolate(x2, size=(x0.size(2), x0.size(3)), mode='bilinear')
        x3 = F.interpolate(x3, size=(x0.size(2), x0.size(3)), mode='bilinear')
        x4 = F.interpolate(x4, size=(x0.size(2), x0.size(3)), mode='bilinear')
        # x_p = torch.cat((x_f_1, x2, x3, x4), 1)
        # x_p = x0 + x1 + x2 + x3 + x4
        x_p = torch.cat((x0, x1, x2, x3, x4),1)
        # x2_p = torch.cat((x1_p, x2), 1)
        # x3_p = torch.cat((x2_p, x3), 1) #[1,256,256,256]
        #post layers
        x_post = F.relu(self.post_bn(self.post_conv(x_p)))

        # First level
        x_b_1 = F.avg_pool2d(x_post, (x_post.size(2) // 128, x_post.size(3) // 128))
        # x_b_1_cat = torch.cat((x_b_1,x_second),1)
        x_b_1 = F.relu(self.pool1_bn(self.pool1_conv(x_b_1))) #[b,64,64,64]
        # Second level
        x_b_2 = F.avg_pool2d(x_post, (x_post.size(2) // 64, x_post.size(3) // 64))
        # x_b_2_cat = torch.cat((x_b_2, x_third), 1)
        x_b_2 = F.relu(self.pool2_bn(self.pool2_conv(x_b_2))) #[b,64,128,128]
        # Third level
        # x_b_3 = F.avg_pool2d(x_post, (x_post.size(2) // 32, x_post.size(3) // 32))
        # x_b_3_cat = torch.cat((x_b_3, x_forth), 1)
        # x_b_3_cat = F.relu(self.pool3_bn(self.pool3_conv(x_b_3_cat)))  # [b,64,128,128]
        # #unsampling layer


        x_b_1_u = F.interpolate(x_b_1, size=(x_post.size(2), x_post.size(3)), mode='bilinear') #[4,64,224,224]
        x_b_2_u = F.interpolate(x_b_2, size=(x_post.size(2), x_post.size(3)), mode='bilinear') #[4,64,224,224]
        # x_b_3_u = F.interpolate(x_b_3_cat, size=(x_post.size(2), x_post.size(3)), mode='bilinear')
        # #concat layer
        # x_c_1 = torch.cat((x_post,x_b_1_u),1)
        # x_c_2 = torch.cat((x_c_1, x_b_2_u), 1)
        x_c_2 = torch.cat((x_post,x_b_1_u,x_b_2_u),1)
        # x_c_3 = torch.cat((x_c_2, x_b_2_u), 1)
        # x_c_4 = torch.cat((x_c_3, x_b_1_u), 1) #[4,384,224,224]
        #domain classifier
        x_p = F.relu(self.bn2(self.conv2(x_c_2)))
        #x = F.avg_pool2d(x_p, (x_p.size(2), x_p.size(3)))
        #x = x.view(-1, 128)
        x = self.conv_cls(x_p)
        # x = F.interpolate(x, size=(224, 224), mode='bilinear')

        return x

if __name__ == "__main__":
    a=[torch.randn([4,18,256,256]),torch.randn([4,18,128,128]),torch.randn([4,18,64,64]),torch.randn([4,18,32,32])]
    dis_model = F_D(input_channel=18)
    x = dis_model(a)
    #/remote-home/share/42/cyc19307140030/yolov5/feature_distill/
    #dis_model.save('/remote-home/share/42/cyc19307140030/yolov5/feature_distill/ddcpp_F_D.pth')
    torch.save(dis_model,'/remote-home/share/42/cyc19307140030/yolov5/feature_distill/ddcpp_F_D.pth')