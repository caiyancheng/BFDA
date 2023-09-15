import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import torch

loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()

m = 'val'
old_path = '/remote-home/share/Cityscapes/leftImg8bit/' + m + '/'
aim_path = '/remote-home/share/Cityscapes/leftImg8bit/' + m + '_front_delta2/'
lb_path = '/remote-home/share/42/cyc19307140030/yolov5/data/cityscapes/labels/'+ m + '_full/'

old_list = os.listdir(old_path)

num = 0

if not os.path.exists(aim_path):
    os.mkdir(aim_path)

for i in old_list:
    sub_dir = old_path+i
    img_list = os.listdir(sub_dir)
    aim_sub_dir = aim_path+i
    if not os.path.exists(aim_sub_dir):
        os.mkdir(aim_sub_dir)
    for j in img_list:
        num += 1
        print(num)
        img_name = sub_dir+'/'+j
        image = Image.open(img_name)
        image = loader(image)
        #local_mean = [image[0].mean(),image[1].mean(),image[2].mean()]
        label_path = lb_path+i+'/'+j.split('.')[0]+'.txt'
        # new_image = torch.zeros(3,1024,2048)
        # a1 = torch.full((1,1024,2048),local_mean[0])
        # a2 = torch.full((1,1024,2048),local_mean[1])
        # a3 = torch.full((1, 1024, 2048), local_mean[2])
        # new_image = torch.cat((a1,a2,a3),0)
        with open(label_path,'r') as fp:
            LB = fp.readlines()
        delta = torch.zeros(1,1024,2048)
        for k in LB:
            x_c = float(k.split('\t')[1])*2048
            y_c = float(k.split('\t')[2])*1024
            w = float(k.split('\t')[3])*2048
            h = float(k.split('\t')[4])*1024
            x_min = max(round(x_c - w/2),0)
            y_min = max(round(y_c - h/2),0)
            x_max = min(round(x_c + w/2),2047)
            y_max = min(round(y_c + h/2),1023)

            delta[0][y_min:y_max, x_min:x_max] = 1
            # new_image[0][y_min:y_max, x_min:x_max] = image[0][y_min:y_max, x_min:x_max]
            # new_image[1][y_min:y_max, x_min:x_max] = image[1][y_min:y_max, x_min:x_max]
            # new_image[2][y_min:y_max, x_min:x_max] = image[2][y_min:y_max, x_min:x_max]
        delta_img = image*delta
        new_image = image*(1-delta)/2+delta_img
        new_image = unloader(new_image)
        # plt.figure()
        # plt.imshow(new_image)
        # plt.show()
        aim_real_path = aim_sub_dir+'/'+j
        # new_image = new_image[..., ::-1]
        cv2.imwrite(aim_real_path, np.array(new_image)[..., ::-1])




