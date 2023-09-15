import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from scipy import io

cityscapes_path_dict = {
    'original_images_path': r'F:\datasets\cityscape\leftImg8bit_trainvaltest\leftImg8bit',
    'original_citypersons_path': r'F:\datasets\CityPersons-master\annotations',
    'original_gtFine_path': r'F:\datasets\cityscape\gtFine_trainvaltest\gtFine',
    'now_images_path': r'F:\BFDA_datasets\cityscapes\images',
    'now_labels_path': r'F:\BFDA_datasets\cityscapes\labels',
}

aim_image_path = r'F:\BFDA_datasets\cityscapes\images'

set_list = ['train', 'val']
mode_list = ['longshortrange_noback_10_20', 'longshortrange_noback_15_25', 'longshortrange_noback_20_30'] #'all' 指的是原图直接复制
mode_num_list = [[1.0,2.0], [1.5,2.5], [2.0,3.0]]
image_size = [2048, 1024]
size_list = [1.0,1.5,2.0,2.5,3.0]

for set in set_list:
    print('Dealing With Set:', set, '\t Please wait patiently')
    image_base_path = os.path.join(cityscapes_path_dict['original_images_path'], set)
    ob_label_base_path = os.path.join(cityscapes_path_dict['now_labels_path'], set)
    image_dir_list = os.listdir(image_base_path)
    for subdir_n in tqdm(range(len(image_dir_list))):
        image_subdir = os.path.join(image_base_path, image_dir_list[subdir_n])
        image_subdir_list = os.listdir(image_subdir)
        for image_name in image_subdir_list:
            source_img_path = os.path.join(image_subdir, image_name)
            source_lb_path = os.path.join(ob_label_base_path, image_dir_list[subdir_n], image_name.replace('.png','.txt'))
            img = cv2.imread(source_img_path)
            mask_white = np.ones_like(img[...,0])
            with open(source_lb_path, 'r') as fp:
                lb_data = fp.readlines()
            mask_bbx_dict = {}
            for size_i in size_list:
                mask_bbx_dict[str(size_i)] = mask_white.copy()
            for ob_id in lb_data:
                ob_data = ob_id.split('\n')[0].split('\t')
                if ob_data[0] == '0':
                    x_center_prop = float(ob_data[1])
                    y_center_prop = float(ob_data[2])
                    w_prop = float(ob_data[3])
                    h_prop = float(ob_data[4])
                    x_left = (x_center_prop - w_prop / 2) * image_size[0]
                    x_right = (x_center_prop + w_prop / 2) * image_size[0]
                    y_up = (y_center_prop - h_prop / 2) * image_size[1]
                    y_down = (y_center_prop + h_prop / 2) * image_size[1]
                    for size_i in size_list:
                        # mask_bbx_dict[str(size_i)][max(round(x_left * size_i), 0):min(round(x_right * size_i), image_size[0]),
                        # max(round(y_up * size_i), 0):min(round(y_down * size_i), image_size[1])] = 0
                        mask_bbx_dict[str(size_i)][
                        max(round((y_center_prop - h_prop * size_i / 2) * image_size[1]), 0):min(
                            round((y_center_prop + h_prop * size_i / 2) * image_size[1]), image_size[1]),
                        max(round((x_center_prop - w_prop * size_i / 2) * image_size[0]), 0):min(
                            round((x_center_prop + w_prop * size_i / 2) * image_size[0]), image_size[0])] = 0
            for mode_i in range(len(mode_list)):
                aim_image_path_mode = os.path.join(aim_image_path, f'{set}_{mode_list[mode_i]}', image_dir_list[subdir_n])
                os.makedirs(aim_image_path_mode, exist_ok=True)
                mode_num = mode_num_list[mode_i]
                aim_imge_mode = img * (1 - mask_bbx_dict[str(mode_num[0])])[...,None] + img * mask_bbx_dict[str(mode_num[1])][...,None]
                cv2.imwrite(os.path.join(aim_image_path_mode, image_name), aim_imge_mode)
                X = 1







