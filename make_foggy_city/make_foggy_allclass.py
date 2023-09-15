import os
import torch
import  xml.dom.minidom as X
import shutil

n_source_pict = 0
n_target_pict = 0
n_source_label = 0
n_target_label = 0

cls_dict = {"person":0,"rider":1,"car":2,"truck":3,"bus":4,"train":5,"motorcycle":6,"bicycle":7}

# source_pict_path = '/remote-home/share/DAOD_Dataset/Cityscapes_multiple_train/JPEGImages/'
source_train_ann_path = '/remote-home/share/DAOD_Dataset/Cityscapes_multiple_train/Annotations/'
target_train_ann_path = '/remote-home/share/DAOD_Dataset/FoggyCityscape_train/Annotations/'
source_val_ann_path = '/remote-home/share/DAOD_Dataset/Cityscapes_multiple_val/Annotations/'
target_val_ann_path = '/remote-home/share/DAOD_Dataset/FoggyCityscape_val/Annotations/'

# source_aim_pict_path = '/remote-home/share/Cityscapes/all_class/cityscapes/images/'
source_aim_label_path = '/remote-home/share/Cityscapes/all_class/cityscapes/labels/'
# target_aim_pict_path = '/remote-home/share/Cityscapes/all_class/foggycityscapes/images/'
target_aim_label_path = '/remote-home/share/Cityscapes/all_class/foggycityscapes/labels/'

# pict_list = os.listdir(pict_path)
# for i in pict_list:
#     t = i.split('_')[0]
#     if t =='source':
#         n_source_pict+=1
#         print(n_source_pict)
#         aim_p = source_aim_pict_path+i
#         shutil.copy(i,aim_p)
#     if t =='target':
#         n_target_pict+=1
#         print(n_target_pict)
#         aim_p = source_aim_pict_path+i
#         shutil.copy(i,aim_p)
#
# print('n_source_pict',n_source_pict)
# print('n_target_pict',n_target_pict)
#####################################################train####################################################
ann_list = os.listdir(source_train_ann_path)
for i in ann_list:
    n_source_label+=1
    print(n_source_label)
    dom = X.parse(source_train_ann_path+i)
    file_name = dom.documentElement.getElementsByTagName('filename')[0].firstChild.data
    dir_name = file_name.split('_')[1]
    real_src_dir_name = source_aim_label_path + 'train/' + dir_name
    real_trg_dir_name = target_aim_label_path + 'train/' + dir_name
    if not os.path.exists(real_src_dir_name):
        os.mkdir(real_src_dir_name)
    if not os.path.exists(real_trg_dir_name):
        os.mkdir(real_trg_dir_name)
    r_name = file_name.split('.')[0].split('_')[1]+'_'+file_name.split('.')[0].split('_')[2]+'_'+file_name.split('.')[0].split('_')[3]+'_'+file_name.split('.')[0].split('_')[4]
    txt_name_src = real_src_dir_name+'/'+r_name+'.txt'
    txt_name_trg = real_trg_dir_name+'/'+r_name+'_foggy_beta_0.02.txt'
    with open(txt_name_src,'w') as fp:
        for j in dom.documentElement.getElementsByTagName('object'):
            cls_name = j.getElementsByTagName('name')[0].firstChild.nodeValue
            bbx = j.getElementsByTagName('bndbox')[0]
            x_min = int(bbx.getElementsByTagName('xmin')[0].firstChild.nodeValue)
            y_min = int(bbx.getElementsByTagName('ymin')[0].firstChild.nodeValue)
            x_max = int(bbx.getElementsByTagName('xmax')[0].firstChild.nodeValue)
            y_max = int(bbx.getElementsByTagName('ymax')[0].firstChild.nodeValue)
            cls_n = cls_dict[cls_name]
            x_center = (x_max+x_min)/(2*2048)
            y_center = (y_max+y_min)/(2*1024)
            w = (x_max-x_min)/2048
            h = (y_max-y_min)/1024
            fp.write(str(cls_n)+'\t'+str(x_center)+'\t'+str(y_center)+'\t'+str(w)+'\t'+str(h)+'\n')
    with open(txt_name_trg,'w') as fp:
        for j in dom.documentElement.getElementsByTagName('object'):
            cls_name = j.getElementsByTagName('name')[0].firstChild.nodeValue
            bbx = j.getElementsByTagName('bndbox')[0]
            x_min = int(bbx.getElementsByTagName('xmin')[0].firstChild.nodeValue)
            y_min = int(bbx.getElementsByTagName('ymin')[0].firstChild.nodeValue)
            x_max = int(bbx.getElementsByTagName('xmax')[0].firstChild.nodeValue)
            y_max = int(bbx.getElementsByTagName('ymax')[0].firstChild.nodeValue)
            cls_n = cls_dict[cls_name]
            x_center = (x_max+x_min)/(2*2048)
            y_center = (y_max+y_min)/(2*1024)
            w = (x_max-x_min)/2048
            h = (y_max-y_min)/1024
            fp.write(str(cls_n)+'\t'+str(x_center)+'\t'+str(y_center)+'\t'+str(w)+'\t'+str(h)+'\n')
################################################################################################################

################################################val##############################################################
ann_list = os.listdir(source_val_ann_path)
for i in ann_list:
    t = i.split('_')[0]
    city_name = i.split('_')[1]
    if t == 'target':
        continue
    if (city_name != 'frankfurt') and (city_name != 'lindau') and (city_name != 'munster'):
        continue
    n_source_label+=1
    print(n_source_label)
    dom = X.parse(source_val_ann_path+i)
    file_name = dom.documentElement.getElementsByTagName('filename')[0].firstChild.data
    dir_name = file_name.split('_')[1]
    real_src_dir_name = source_aim_label_path + 'val/' + dir_name
    real_trg_dir_name = target_aim_label_path + 'val/' + dir_name
    if not os.path.exists(real_src_dir_name):
        os.mkdir(real_src_dir_name)
    if not os.path.exists(real_trg_dir_name):
        os.mkdir(real_trg_dir_name)
    r_name = file_name.split('.')[0].split('_')[1]+'_'+file_name.split('.')[0].split('_')[2]+'_'+file_name.split('.')[0].split('_')[3]+'_'+file_name.split('.')[0].split('_')[4]
    txt_name_src = real_src_dir_name+'/'+r_name+'.txt'
    txt_name_trg = real_trg_dir_name+'/'+r_name+'_foggy_beta_0.02.txt'
    with open(txt_name_src,'w') as fp:
        for j in dom.documentElement.getElementsByTagName('object'):
            cls_name = j.getElementsByTagName('name')[0].firstChild.nodeValue
            bbx = j.getElementsByTagName('bndbox')[0]
            x_min = int(bbx.getElementsByTagName('xmin')[0].firstChild.nodeValue)
            y_min = int(bbx.getElementsByTagName('ymin')[0].firstChild.nodeValue)
            x_max = int(bbx.getElementsByTagName('xmax')[0].firstChild.nodeValue)
            y_max = int(bbx.getElementsByTagName('ymax')[0].firstChild.nodeValue)
            cls_n = cls_dict[cls_name]
            x_center = (x_max+x_min)/(2*2048)
            y_center = (y_max+y_min)/(2*1024)
            w = (x_max-x_min)/2048
            h = (y_max-y_min)/1024
            fp.write(str(cls_n)+'\t'+str(x_center)+'\t'+str(y_center)+'\t'+str(w)+'\t'+str(h)+'\n')
    with open(txt_name_trg,'w') as fp:
        for j in dom.documentElement.getElementsByTagName('object'):
            cls_name = j.getElementsByTagName('name')[0].firstChild.nodeValue
            bbx = j.getElementsByTagName('bndbox')[0]
            x_min = int(bbx.getElementsByTagName('xmin')[0].firstChild.nodeValue)
            y_min = int(bbx.getElementsByTagName('ymin')[0].firstChild.nodeValue)
            x_max = int(bbx.getElementsByTagName('xmax')[0].firstChild.nodeValue)
            y_max = int(bbx.getElementsByTagName('ymax')[0].firstChild.nodeValue)
            cls_n = cls_dict[cls_name]
            x_center = (x_max+x_min)/(2*2048)
            y_center = (y_max+y_min)/(2*1024)
            w = (x_max-x_min)/2048
            h = (y_max-y_min)/1024
            fp.write(str(cls_n)+'\t'+str(x_center)+'\t'+str(y_center)+'\t'+str(w)+'\t'+str(h)+'\n')
# ann_list = os.listdir(source_val_ann_path)
# for i in ann_list:
#     t = i.split('_')[0]
#     city_name = i.split('_')[1]
#     if t == 'target':
#         continue
#     if (city_name != 'frankfurt') and (city_name != 'lindau') and (city_name != 'munster'):
#         continue
#     n_source_label+=1
#     print(n_source_label)
#     dom = X.parse(source_val_ann_path+i)
#     file_name = dom.documentElement.getElementsByTagName('filename')[0].firstChild.data
#     txt_name = file_name.split('.')[0]+'.txt'
#     with open(source_aim_label_path+'val/'+txt_name,'w') as fp:
#         for j in dom.documentElement.getElementsByTagName('object'):
#             cls_name = j.getElementsByTagName('name')[0].firstChild.nodeValue
#             bbx = j.getElementsByTagName('bndbox')[0]
#             x_min = int(bbx.getElementsByTagName('xmin')[0].firstChild.nodeValue)
#             y_min = int(bbx.getElementsByTagName('ymin')[0].firstChild.nodeValue)
#             x_max = int(bbx.getElementsByTagName('xmax')[0].firstChild.nodeValue)
#             y_max = int(bbx.getElementsByTagName('ymax')[0].firstChild.nodeValue)
#             cls_n = cls_dict[cls_name]
#             x_center = (x_max+x_min)/(2*2048)
#             y_center = (y_max+y_min)/(2*1024)
#             w = (x_max-x_min)/2048
#             h = (y_max-y_min)/1024
#             fp.write(str(cls_n)+'\t'+str(x_center)+'\t'+str(y_center)+'\t'+str(w)+'\t'+str(h)+'\n')

