import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import torch

loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()

real_img_path = '/remote-home/share/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000001_013016_leftImg8bit.png'
img_path = '/remote-home/share/Cityscapes/leftImg8bit/val_remove_nearback_semantic/frankfurt/frankfurt_000001_013016_leftImg8bit.png'
aim_path = '/remote-home/share/Cityscapes/leftImg8bit/demo/frankfurt_000001_013016_leftImg8bit_inner_back.png'

image = Image.open(img_path)
real_img = Image.open(real_img_path)
image = loader(image)
real_img = loader(real_img)

black_area = real_img-image
black_onc = (black_area[0]+black_area[1]+black_area[2]).unsqueeze(0)
bal = torch.where(black_onc>0, torch.ones(1,1024,2048), torch.zeros(1,1024,2048))
# red_area = torch.cat((bal,torch.zeros(1,1024,2048),torch.zeros(1,1024,2048)),0)

image[2] = real_img[2] + (bal/8.).squeeze(0)-bal*real_img[2]/2.

image = unloader(image)
plt.figure()
plt.imshow(image)
plt.show()

