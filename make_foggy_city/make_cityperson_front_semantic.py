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
semantic_path = '/remote-home/share/Cityscapes/gtFine/'+m+'/'#gtFine
bbx_path = '/remote-home/share/Cityscapes/citypersons/'+m+'_full/'
aim_path = '/remote-home/share/Cityscapes/leftImg8bit/'+m+'_remove_nearback_semantic_delta2_1.3/'
#aim_path = '/remote-home/share/Cityscapes/leftImg8bit/'+m+'_far_back/'

if not os.path.exists(aim_path):
    os.mkdir(aim_path)

pict_dir = os.listdir(pict_path)

for i in pict_dir:#子目录
    sub_path = pict_path+i
    img_list = os.listdir(sub_path)
    aim_dir = aim_path+i
    if not os.path.exists(aim_dir):
        os.mkdir(aim_dir)
    for j in img_list: #aachen_000000_000019_leftImg8bit  #aachen_000004_000019_gtFine_labelIds
        num+=1
        print(num)
        im = j.split('.')[0]
        semantic_label = im.split('_')[0]+'_'+im.split('_')[1]+'_'+im.split('_')[2]+'_gtFine_labelIds.png'
        real_img_path = pict_path+i+'/'+j
        real_label_path = semantic_path+i+'/'+semantic_label
        real_bbx_path = bbx_path+i+'/'+im+'.txt'
        img = Image.open(real_img_path) #[3,1024,2048]
        lb = Image.open(real_label_path) #[1,1024,2048]
        with open(real_bbx_path,'r') as fp:
            data = fp.readlines()
        lb_bbx_back = torch.ones(1,1024,2048)
        for k in data:
            x_c = float(k.split('\t')[1])*2048
            y_c = float(k.split('\t')[2])*1024
            w = float(k.split('\t')[3])*2048*1.3
            h = float(k.split('\t')[4])*1024*1.3
            x_min = max(round(x_c - w/2),0)
            x_max = min(round(x_c + w/2),2047)
            y_min = max(round(y_c - h/2),0)
            y_max = min(round(y_c + h/2),1023)
            lb_bbx_back[0][y_min:y_max, x_min:x_max] = 0
        img = loader(img)
        lb = (loader(lb)*255).int()
        b = torch.zeros(1,1024,2048).int()
        new_lb = torch.where(lb==24,lb,b)/24
        # local_mean = [0,0,0]
        # local_mean[0] = (img[0] * new_lb).sum() / new_lb.sum()
        # local_mean[1] = (img[1] * new_lb).sum() / new_lb.sum()
        # local_mean[2] = (img[2] * new_lb).sum() / new_lb.sum()
        #new_lb = 1 - torch.where(lb == 24, lb, b) / 24
        and_lb = new_lb+lb_bbx_back
        new_lb = torch.where(and_lb == 2, torch.ones(1,1024,2048), and_lb)
        #black_lb = 1-new_lb
        # local_mean = [img[0].mean(), img[1].mean(), img[2].mean()]
        # a1 = torch.full((1, 1024, 2048), local_mean[0])
        # a2 = torch.full((1, 1024, 2048), local_mean[1])
        # a3 = torch.full((1, 1024, 2048), local_mean[2])
        # back_ground = torch.cat((a1, a2, a3), 0)
        # back_ground = back_ground * new_lb
        #new_lb = lb_bbx_back
        img = img*new_lb + img*(1-new_lb)/2
        new_image = unloader(img)
        # plt.figure()
        # plt.imshow(new_image)
        # plt.show()
        aim_real_path = aim_dir + '/' + j
        cv2.imwrite(aim_real_path, np.array(new_image)[..., ::-1])