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
# mode_list = ['no_foreground', 'no_short_range_background', 'no_long_range_background'] #'all' 指的是原图直接复制
image_size = [2048, 1024]

for set in set_list:
    print('Dealing With Set:', set, '\t Please wait patiently')
    image_base_path = os.path.join(cityscapes_path_dict['original_images_path'], set)
    bbx_label_base_path = os.path.join(cityscapes_path_dict['now_labels_path'], set)
    segmentation_label_base_path = os.path.join(cityscapes_path_dict['original_gtFine_path'], set)

    image_dir_list = os.listdir(image_base_path)
    for subdir_n in tqdm(range(len(image_dir_list))):
        image_subdir = os.path.join(image_base_path, image_dir_list[subdir_n])
        image_subdir_list = os.listdir(image_subdir)
        for image_name in image_subdir_list:
            source_img_path = os.path.join(image_subdir, image_name)
            source_bbx_lb_path = os.path.join(bbx_label_base_path, image_dir_list[subdir_n], image_name.replace('.png','.txt'))
            source_segmentation_lb_path = os.path.join(segmentation_label_base_path, image_dir_list[subdir_n], image_name.split('.')[0].replace('_leftImg8bit', '_gtFine_labelIds.png'))
            img = cv2.imread(source_img_path)
            img_segmentation = cv2.imread(source_segmentation_lb_path, cv2.IMREAD_GRAYSCALE)
            with open(source_bbx_lb_path, 'r') as fp:
                lb_data = fp.readlines()
            mask_bbx = np.ones_like(img[...,0])
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
                    mask_bbx[
                    max(round((y_center_prop - h_prop / 2) * image_size[1]), 0):min(
                        round((y_center_prop + h_prop / 2) * image_size[1]), image_size[1]),
                    max(round((x_center_prop - w_prop / 2) * image_size[0]), 0):min(
                        round((x_center_prop + w_prop / 2) * image_size[0]), image_size[0])] = 0
            mask_foreground = img_segmentation.copy()
            # if image_name == 'aachen_000020_000019_leftImg8bit.png':
            #     X = 1
            mask_foreground[mask_foreground != 24] = 0
            mask_foreground[mask_foreground == 24] = 1

            aim_image_path_mode_1 = os.path.join(aim_image_path, f'{set}_no_foreground', image_dir_list[subdir_n])
            os.makedirs(aim_image_path_mode_1, exist_ok=True)
            aim_image_mode = img * (1 - mask_foreground)[...,None]
            cv2.imwrite(os.path.join(aim_image_path_mode_1, image_name), aim_image_mode)

            aim_image_path_mode_2 = os.path.join(aim_image_path, f'{set}_no_short_range_background', image_dir_list[subdir_n])
            os.makedirs(aim_image_path_mode_2, exist_ok=True)
            aim_image_mode = img * mask_bbx[...,None] + img * mask_foreground[...,None]
            cv2.imwrite(os.path.join(aim_image_path_mode_2, image_name), aim_image_mode)

            aim_image_path_mode_3 = os.path.join(aim_image_path, f'{set}_no_long_range_background', image_dir_list[subdir_n])
            os.makedirs(aim_image_path_mode_3, exist_ok=True)
            aim_image_mode = img * (1 - mask_bbx)[...,None]
            cv2.imwrite(os.path.join(aim_image_path_mode_3, image_name), aim_image_mode)

            aim_image_path_mode_4 = os.path.join(aim_image_path, f'{set}_no_all_background',
                                                 image_dir_list[subdir_n])
            os.makedirs(aim_image_path_mode_4, exist_ok=True)
            aim_image_mode = img * mask_foreground[...,None]
            cv2.imwrite(os.path.join(aim_image_path_mode_4, image_name), aim_image_mode)









