import os
from PIL import Image
import torch
import cv2
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()

m = 'val'

num = 0

pict_path = '/remote-home/share/Cityscapes/leftImg8bit/'+m+'/'
bbx_path = '/remote-home/share/Cityscapes/citypersons/'+m+'_full/'
aim_path = '/remote-home/share/Cityscapes/leftImg8bit/'+m+'_ls0.0_1.0/'

k_scale_S = 0.0
k_scale_L = 1.0

if not os.path.exists(aim_path):
    os.mkdir(aim_path)

pict_dir = os.listdir(pict_path)
for i in pict_dir:
    sub_path = pict_path + i
    img_list = os.listdir(sub_path)
    aim_dir = aim_path + i
    if not os.path.exists(aim_dir):
        os.mkdir(aim_dir)
    for j in img_list:
        num+=1
        print(num)
        im = j.split('.')[0]
        real_img_path = pict_path + i + '/' + j
        real_bbx_path = bbx_path + i + '/' + im + '.txt'
        img = Image.open(real_img_path)  # [3,1024,2048]
        with open(real_bbx_path,'r') as fp:
            data = fp.readlines()
        lb_bbx_back = torch.ones(1, 1024, 2048)
        for k in data:
            x_c = float(k.split('\t')[1])*2048
            y_c = float(k.split('\t')[2])*1024
            w = float(k.split('\t')[3])*2048
            h = float(k.split('\t')[4])*1024

            x_min_2 = max(round(x_c - w * k_scale_L / 2), 0)
            x_max_2 = min(round(x_c + w * k_scale_L / 2), 2047)
            y_min_2 = max(round(y_c - h * k_scale_L / 2), 0)
            y_max_2 = min(round(y_c + h * k_scale_L / 2), 1023)

            lb_bbx_back[0][y_min_2:y_max_2, x_min_2:x_max_2] = 0
        for k in data:
            x_c = float(k.split('\t')[1])*2048
            y_c = float(k.split('\t')[2])*1024
            w = float(k.split('\t')[3])*2048
            h = float(k.split('\t')[4])*1024
            x_min_1 = max(round(x_c - w * k_scale_S / 2), 0)
            x_max_1 = min(round(x_c + w * k_scale_S / 2), 2047)
            y_min_1 = max(round(y_c - h * k_scale_S / 2), 0)
            y_max_1 = min(round(y_c + h * k_scale_S / 2), 1023)

            lb_bbx_back[0][y_min_1:y_max_1, x_min_1:x_max_1] = 1

        img = loader(img)
        img = img*lb_bbx_back
        new_image = unloader(img)
        aim_real_path = aim_dir + '/' + j
        cv2.imwrite(aim_real_path, np.array(new_image)[..., ::-1])