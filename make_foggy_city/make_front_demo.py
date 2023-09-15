import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import torch

loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()

img_path = '/remote-home/share/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000001_013016_leftImg8bit.png'
aim_path = '/remote-home/share/Cityscapes/leftImg8bit/demo/frankfurt_000001_013016_leftImg8bit_front.png'
lb_path = '/remote-home/share/42/cyc19307140030/yolov5/data/cityscapes/labels/val_full/frankfurt/frankfurt_000001_013016_leftImg8bit.txt'

image = Image.open(img_path)
image = loader(image)
back_image = torch.cat((torch.ones(1,1024,2048),torch.zeros(1,1024,2048),torch.zeros(1,1024,2048)),0)#RGB
new_image = image/2. + back_image/4.#'#/60. #new_image.clone()/2.+
with open(lb_path,'r') as fp:
    LB = fp.readlines()
for k in LB:
    x_c = float(k.split('\t')[1]) * 2048
    y_c = float(k.split('\t')[2]) * 1024
    w = float(k.split('\t')[3]) * 2048
    h = float(k.split('\t')[4]) * 1024
    x_min = max(round(x_c - w / 2), 0)
    y_min = max(round(y_c - h / 2), 0)
    x_max = min(round(x_c + w / 2), 2047)
    y_max = min(round(y_c + h / 2), 1023)

    new_image[0][y_min:y_max, x_min:x_max] = image[0][y_min:y_max, x_min:x_max]
    new_image[1][y_min:y_max, x_min:x_max] = image[1][y_min:y_max, x_min:x_max]
    new_image[2][y_min:y_max, x_min:x_max] = image[2][y_min:y_max, x_min:x_max]

# new_image = unloader(new_image)
# cv2.imwrite(aim_path, np.array(new_image)[..., ::-1])
new_image = unloader(new_image)
plt.figure()
plt.imshow(new_image)
plt.show()
# image = unloader(image)
# cv2.imwrite(aim_path, np.array(image)[..., ::-1])





