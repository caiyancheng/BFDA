import torch
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import models
from torchvision import transforms
import cv2
import time

size = (640,320)
# 加载deeplabv3模型进行语义分割
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model = model.eval()

img = cv2.imread(r'/remote-home/caiyancheng/BFDA_datasets/cityscapes/images/val_all/frankfurt/frankfurt_000001_057181_leftImg8bit.png')
img = cv2.resize(img, size)
img = torch.tensor(img).permute(2,0,1)[None,...]/255

start_time = time.time()
output = model(img)
end_time = time.time()
time_using = start_time - end_time
print(time_using)
print(output['out'].shape)
output = torch.argmax(output['out'].squeeze(), dim=0).detach().cpu().numpy()

output[output==15] = 255
output[output!=255] = 0

cv2.imwrite(f'/remote-home/caiyancheng/outlabel_frankfurt_000001_057181_leftImg8bit_size_{size[0]}_size{size[1]}_time_{time_using}.png', np.array(output))