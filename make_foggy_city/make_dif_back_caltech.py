import os
from PIL import Image
import torch
import cv2
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import shutil

loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()

a1 = 0 #0.99~1.0
a2 = 0 #0.96~0.99
a3 = 0 #0~0.96
# # a4 = 0 #0.99~1
# # a5 = 0 #0.96~0.99
# # a6 = 0 #0~0.96
num = 0
city_path = '/remote-home/share/Caltech/images/test/'
city_dir = os.listdir(city_path)
for i in city_dir:
    subpath = city_path+i
    subdir = os.listdir(subpath)
    for j in subdir:
        num+=1
        print(num)
        real_path = subpath+'/'+j
        img = Image.open(real_path)
        img = loader(img)# [3,1024,2048]
        zero_num = np.sum((img[0]+img[1]+img[2]).numpy()==0)
        proportion = zero_num/(img.shape[1]*img.shape[2])
        if(proportion>0.99):
            a1+=1
            new_img_dir = '/remote-home/share/42/cyc19307140030/CVPR/BFNet/Caltech_dif_back/0.99_1/images/' + i
            new_lb_dir = '/remote-home/share/42/cyc19307140030/CVPR/BFNet/Caltech_dif_back/0.99_1/labels/' + i
        elif(proportion>0.96):
            a2+=1
            new_img_dir = '/remote-home/share/42/cyc19307140030/CVPR/BFNet/Caltech_dif_back/0.96_0.99/images/' + i
            new_lb_dir = '/remote-home/share/42/cyc19307140030/CVPR/BFNet/Caltech_dif_back/0.96_0.99/labels/' + i
        else:
            a3+=1
            new_img_dir = '/remote-home/share/42/cyc19307140030/CVPR/BFNet/Caltech_dif_back/0_0.96/images/' + i
            new_lb_dir = '/remote-home/share/42/cyc19307140030/CVPR/BFNet/Caltech_dif_back/0_0.96/labels/' + i
        # if not os.path.exists(new_img_dir):
        #     os.mkdir(new_img_dir)
        # if not os.path.exists(new_lb_dir):
        #     os.mkdir(new_lb_dir)
        # new_img_path = new_img_dir+'/'+j.replace('leftImg8bit.png','leftImg8bit_foggy_beta_0.02.png')
        # new_lb_path = new_lb_dir+'/'+j.replace('leftImg8bit.png','leftImg8bit_foggy_beta_0.02.txt')
        # olg_img_path = '/remote-home/share/Cityscapes/newfoggy_cyc/images/0.02/val/'+ i + '/' + j.replace('leftImg8bit.png','leftImg8bit_foggy_beta_0.02.png')
        # olg_lb_path = '/remote-home/share/42/cyc19307140030/yolov5/data/foggycityscapes/labels/val_0.02/' + i + '/' + j.replace('leftImg8bit.png','leftImg8bit_foggy_beta_0.02.txt')
        # shutil.copy(olg_lb_path,new_lb_path)
        # shutil.copy(olg_img_path,new_img_path)
print('a1:',a1)
print('a2:',a2)
print('a3:',a3)

