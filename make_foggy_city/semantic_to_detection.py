import json
import os
import shutil
from os import listdir, getcwd
from os.path import join
import os.path
import os
from PIL import Image
import torch
import cv2
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
num = 0
wrong_num = 0
loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()

m = 'val'

pict_path = '/remote-home/share/BDD100K/bdd100k_images/bdd100k/images/10k/' + m + '/'
aim_pict_path = '/remote-home/share/BDD100k_onlyperson/BDD10k/images/' + m + '/'
semantic_lb_path = '/remote-home/share/BDD100K/bdd100k/labels/sem_seg/polygons/' + 'sem_seg_' + m + '.json'
aim_lb_path = '/remote-home/share/BDD100k_onlyperson/BDD10k/labels/' + m + '/'
if not os.path.exists(aim_lb_path):
    os.mkdir(aim_lb_path)
if not os.path.exists(aim_pict_path):
    os.mkdir(aim_pict_path)


def position(pos): #该函数用来找出xmin,ymin,xmax,ymax即bbox包围框
    x = []
    y = []
    nums = len(pos)
    for i in range(nums):
        x.append(pos[i][0])
        y.append(pos[i][1])
    x_max = min(max(x),1279)
    x_min = max(min(x),0)
    y_max = min(max(y),719)
    y_min = max(min(y),0)
    b = (float(x_min), float(x_max), float(y_min), float(y_max))
    return b

with open(semantic_lb_path, 'r') as fp1:
    data_se = json.load(fp1)

for i in data_se:
    num+=1
    print(num)
    pict_name = i['name']
    real_pict_name = pict_path + i['name']
    try:
        img = Image.open(real_pict_name)  # [3,1024,2048]
    except:
        wrong_num+=1
        continue
    img = loader(img)
    if img.shape[1]!= 720 or img.shape[2]!= 1280:
        wrong_num += 1
        continue
    real_aim_pict_path = aim_pict_path+i['name']
    shutil.copy(real_pict_name,real_aim_pict_path)
    txt_name = i['name'].split('.')[0] + '.txt'
    real_txt_name = aim_lb_path+txt_name
    lb = i['labels']
    with open(real_txt_name,'w') as fp2:
        for j in lb:
            if j['category'] == 'person':
                poly_list = j['poly2d'][0]['vertices']
                bbx = position(poly_list) #x_min,x_max,y_min,y_max
                x_center = (bbx[0] + bbx[1]) / (2 * 1280)
                y_center = (bbx[2] + bbx[3]) / (2 * 720)
                w = (bbx[1] - bbx[0]) / 1280
                h = (bbx[3] - bbx[2]) / 720
                fp2.write('0\t'+str(x_center)+'\t'+str(y_center)+'\t'+str(w)+'\t'+str(h)+'\n')
print(wrong_num)