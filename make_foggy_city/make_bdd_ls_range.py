import os
from PIL import Image
import torch
import cv2
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()


num = 0

w_w = 1280
h_h = 720

pict_path = '/remote-home/share/BDD100k_onlyperson/BDD10k/images/val/'
bbx_path = '/remote-home/share/BDD100k_onlyperson/BDD10k/labels/val/'
aim_path = '/remote-home/share/BDD100k_onlyperson/BDD10k/images/val_ls_2.0_3.0/'

s_r = 2.0
l_r = 3.0

if not os.path.exists(aim_path):
    os.mkdir(aim_path)

pict_dir = os.listdir(pict_path)

for i in pict_dir:#子目录
    num+=1
    print(num)
    im = i.split('.')[0]
    semantic_label = im +'.png'
    real_img_path = pict_path + i
    real_bbx_path = bbx_path + im + '.txt'
    img = Image.open(real_img_path)  # [3,1024,2048]
    with open(real_bbx_path, 'r') as fp:
        data = fp.readlines()
    lb_bbx_back = torch.ones(1, h_h, w_w)
    for k in data:
        x_c = float(k.split('\t')[1]) * w_w
        y_c = float(k.split('\t')[2]) * h_h
        w = float(k.split('\t')[3]) * w_w
        h = float(k.split('\t')[4]) * h_h
        x_min = max(round(x_c - w * l_r / 2), 0)
        x_max = min(round(x_c + w * l_r / 2), w_w-1)
        y_min = max(round(y_c - h * l_r / 2), 0)
        y_max = min(round(y_c + h * l_r / 2), h_h-1)
        lb_bbx_back[0][y_min:y_max, x_min:x_max] = 0
    for k in data:
        x_c = float(k.split('\t')[1]) * w_w
        y_c = float(k.split('\t')[2]) * h_h
        w = float(k.split('\t')[3]) * w_w
        h = float(k.split('\t')[4]) * h_h
        x_min = max(round(x_c - w * s_r / 2), 0)
        x_max = min(round(x_c + w * s_r / 2), w_w-1)
        y_min = max(round(y_c - h * s_r / 2), 0)
        y_max = min(round(y_c + h * s_r / 2), h_h-1)
        lb_bbx_back[0][y_min:y_max, x_min:x_max] = 1
    img = loader(img)
    img = lb_bbx_back*img
    new_image = unloader(img)
    # plt.figure()
    # plt.imshow(new_image)
    # plt.show()
    aim_real_path = aim_path + i
    cv2.imwrite(aim_real_path, np.array(new_image)[..., ::-1])