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
semantic_path = '/remote-home/share/BDD100K/bdd100k/labels/sem_seg/masks/val/'
bbx_path = '/remote-home/share/BDD100k_onlyperson/BDD10k/labels/val/'
aim_path = '/remote-home/share/BDD100k_onlyperson/BDD10k/images/val_no_allback/'

if not os.path.exists(aim_path):
    os.mkdir(aim_path)

pict_dir = os.listdir(pict_path)

for i in pict_dir:#子目录
    if i == '7dc08598-f42e2015.jpg':
        PP = 1
    num+=1
    print(num)
    im = i.split('.')[0]
    semantic_label = im +'.png'
    real_img_path = pict_path + i
    real_label_path = semantic_path + semantic_label
    real_bbx_path = bbx_path + im + '.txt'
    img = Image.open(real_img_path)  # [3,1024,2048]
    lb = Image.open(real_label_path)  # [1,1024,2048]
    with open(real_bbx_path, 'r') as fp:
        data = fp.readlines()
    lb_bbx_back = torch.ones(1, h_h, w_w)
    for k in data:
        x_c = float(k.split('\t')[1]) * w_w
        y_c = float(k.split('\t')[2]) * h_h
        w = float(k.split('\t')[3]) * w_w
        h = float(k.split('\t')[4]) * h_h
        x_min = max(round(x_c - w / 2), 0)
        x_max = min(round(x_c + w / 2), w_w-1)
        y_min = max(round(y_c - h / 2), 0)
        y_max = min(round(y_c + h / 2), h_h-1)
        lb_bbx_back[0][y_min:y_max, x_min:x_max] = 0
    img = loader(img)
    lb = (loader(lb) * 255).int()
    b = torch.zeros(1, h_h, w_w).int()
    new_lb = torch.where(lb == 11, lb, b) / 11
    # local_mean = [0,0,0]
    # local_mean[0] = (img[0] * new_lb).sum() / new_lb.sum()
    # local_mean[1] = (img[1] * new_lb).sum() / new_lb.sum()
    # local_mean[2] = (img[2] * new_lb).sum() / new_lb.sum()
    # new_lb = 1 - torch.where(lb == 24, lb, b) / 24
    # and_lb = new_lb + lb_bbx_back
    # new_lb = torch.where(and_lb == 2, torch.ones(1, 1024, 2048), and_lb)
    # black_lb = 1-new_lb
    # local_mean = [img[0].mean(), img[1].mean(), img[2].mean()]
    # a1 = torch.full((1, 1024, 2048), local_mean[0])
    # a2 = torch.full((1, 1024, 2048), local_mean[1])
    # a3 = torch.full((1, 1024, 2048), local_mean[2])
    # back_ground = torch.cat((a1, a2, a3), 0)
    # back_ground = back_ground * new_lb
    # new_lb = lb_bbx_back
    # img = img * new_lb + img * (1 - new_lb) / 2
    #############nofarback
    # img = (1-lb_bbx_back)*img
    #############

    #############nonearback
    # and_lb = new_lb + lb_bbx_back
    # new_lb = torch.where(and_lb == 2, torch.ones(1, h_h, w_w), and_lb)
    # img = new_lb * img
    #############

    ######################nofront
    # new_lb = 1 - new_lb
    # img = new_lb * img
    ######################

    ###########################noallback
    img = new_lb * img
    #######################
    new_image = unloader(img)
    # plt.figure()
    # plt.imshow(new_image)
    # plt.show()
    aim_real_path = aim_path + i
    cv2.imwrite(aim_real_path, np.array(new_image)[..., ::-1])