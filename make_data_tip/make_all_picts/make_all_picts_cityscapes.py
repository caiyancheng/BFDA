import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm
from scipy import io

cityscapes_path_dict = {
    'original_images_path': r'F:\datasets\cityscape\leftImg8bit_trainvaltest\leftImg8bit',
    'original_citypersons_path': r'F:\datasets\CityPersons-master\annotations',
    'original_gtFine_path': r'F:\datasets\cityscape\gtFine_trainvaltest\gtFine'
}

aim_image_path = r'F:\BFDA_datasets\cityscapes\images'
aim_label_path = r'F:\BFDA_datasets\cityscapes\labels'

set_list = ['train', 'val']

pict_num_dict = [2975, 500]
image_size = [2048, 1024]
########转移图片
# for set in set_list:
#     print('Dealing With Set:', set, '\t Please wait patiently')
#     image_base_path = os.path.join(cityscapes_path_dict['original_images_path'], set)
#     ob_label_base_path = os.path.join(cityscapes_path_dict['original_citypersons_path'], f'anno_{set}.mat')
#     image_dir_list = os.listdir(image_base_path)
#     # mat_labels = io.loadmat(ob_label_base_path)[f'anno_{set}_aligned'][0]
#     for subdir_n in tqdm(range(len(image_dir_list))):
#         image_subdir = os.path.join(image_base_path, image_dir_list[subdir_n])
#         image_subdir_list = os.listdir(image_subdir)
#         for image_name in image_subdir_list:
#             source_img_path = os.path.join(image_subdir, image_name)
#             aim_img_dir = os.path.join(aim_image_path, f'{set}_all', image_dir_list[subdir_n])
#             os.makedirs(aim_img_dir, exist_ok=True)
#             aim_img_path = os.path.join(aim_img_dir, image_name)
#             shutil.copy(source_img_path, aim_img_path)

########转移标签 #class, x_center/x, y_center/y, w/x, h/y
for set in set_list:
    print('Dealing With Set:', set, '\t Please wait patiently')
    ob_label_base_path = os.path.join(cityscapes_path_dict['original_citypersons_path'], f'anno_{set}.mat')
    mat_labels = io.loadmat(ob_label_base_path)[f'anno_{set}_aligned'][0]
    for label_num in tqdm(range(mat_labels.shape[0])):
        whole_data = mat_labels[label_num][0][0]
        label_subpath = str(whole_data[0][0])
        label_name = str(whole_data[1][0]).split('.')[0]
        label_data = whole_data[2]
        label_txt_dir_name = os.path.join(aim_label_path, set, label_subpath)
        label_txt_file_name = os.path.join(label_txt_dir_name, f'{label_name}.txt')
        os.makedirs(label_txt_dir_name, exist_ok=True)
        with open(label_txt_file_name, 'w') as fp:
            for obj_data in label_data:
                if obj_data[0] != 1:
                    continue
                else:
                    x_1 = obj_data[1]
                    y_1 = obj_data[2]
                    w = obj_data[3]
                    h = obj_data[4]
                    x_center_prop = (x_1 + w / 2) / image_size[0]
                    y_center_prop = (y_1 + h / 2) / image_size[1]
                    w_prop = w / image_size[0]
                    h_prop = h / image_size[1]
                    fp.write(f'0\t{x_center_prop}\t{y_center_prop}\t{w_prop}\t{h_prop}\n')







