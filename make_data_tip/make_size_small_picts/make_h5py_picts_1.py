import os

import torch
import numpy as np
import pandas
import h5py

images_path = r'F:\BFDA_datasets\cityscapes\images\train_all'
labels_path = r'F:\BFDA_datasets\cityscapes\labels\train'

hf = h5py.File(r'F:\BFDA_datasets\cityscapes\h5py/train_all_h5py.h5', 'w', driver='core')


hf.create_dataset()

#我发现好像用不了....
