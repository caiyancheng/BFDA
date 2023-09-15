import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import torch

loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()

old_path = '/remote-home/share/42/cyc19307140030/yolov5/runs/trainforpaper_crossdomain_transformer_debug/exp68/src_img_aachen_000031_000019_background_prob_mask_src.png'
aim_path = '/remote-home/share/42/cyc19307140030/yolov5/debug/show/F/front.png'

image = Image.open(old_path)
image = loader(image)
front_img = 1 - image
front_img = unloader(front_img)

cv2.imwrite(aim_path, np.array(front_img)[..., ::-1])
