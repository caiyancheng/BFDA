import torch
import torch.nn as nn
from utils.general import *
# import vit_pytorch
#from vit_pytorch.cvt import CvT
from cvt import CvT
import cvt_official
import cvt_simple
import cvt_simple_18
import cvt_hr
import cvt_hr_18
import cvt_hr_cat
import cvt_hr_bce
import cvt_official_18
from torch.cuda import amp
from domain_ad import *
import matplotlib.pyplot as plt
from torchvision import *
from einops import rearrange
import json
import torch.nn.functional as F
import random


unloader = transforms.ToPILImage()
loader = transforms.Compose([transforms.ToTensor()])

def feature_cut(feature = None,device = None, H = 4,L = 3):
    for bi in range(len(feature[0])):
        for bj in range(4):
            for k in range(feature[bj][bi].shape[0]):
                max_f = float(feature[bj][bi][k].max())
                min_f = float(feature[bj][bi][k].min())
                mean_f = float(feature[bj][bi][k].mean())
                max_f_full = torch.full(feature[bj][bi][k].shape, (max_f - mean_f) / H + mean_f).to(device)
                min_f_full = torch.full(feature[bj][bi][k].shape, (min_f - mean_f) / L + mean_f).to(device)
                mean_f_full = torch.full(feature[bj][bi][k].shape, mean_f).to(device)
                # feature_afterconv_src[bj][bi] = torch.where(feature_afterconv_src[bj][bi] - mean_f > (max_f-mean_f)/3, max_f_full, feature_afterconv_src[bj][bi])
                # feature_afterconv_src[bj][bi] = torch.where(feature_afterconv_src[bj][bi] - mean_f < (min_f-mean_f)/1.3, min_f_full, feature_afterconv_src[bj][bi])
                feature[bj][bi][k] = torch.where(feature[bj][bi][k] - mean_f > (max_f - mean_f) / H, mean_f_full,
                                              feature[bj][bi][k])
                feature[bj][bi][k] = torch.where(feature[bj][bi][k] - mean_f < (min_f - mean_f) / L, mean_f_full,
                                              feature[bj][bi][k])

    return feature

def paste_label(imgs,real_labels, all_black=False, radar = False):
    for di in range(len(real_labels)):
        img_num = int(real_labels[di][0])
        x_center = round(float(real_labels[di][2] * imgs.shape[3]))
        y_center = round(float(real_labels[di][3] * imgs.shape[2]))
        wide = round(float(real_labels[di][4] * imgs.shape[3]))
        height = round(float(real_labels[di][5] * imgs.shape[2]))
        x_min = round(max(x_center - wide / 2, 0))
        x_max = round(min(x_center + wide / 2, imgs.shape[3] - 1))
        y_min = round(max(y_center - height / 2, 0))
        y_max = round(min(y_center + height / 2, imgs.shape[2] - 1))
        if all_black:
            imgs[img_num][0][y_min:y_max, x_min:x_max] = 0
            imgs[img_num][1][y_min:y_max, x_min:x_max] = 0
            imgs[img_num][2][y_min:y_max, x_min:x_max] = 0
        else:
            imgs[img_num][0][y_min:y_max, x_min:x_max] = imgs[img_num][0].mean()
            imgs[img_num][1][y_min:y_max, x_min:x_max] = imgs[img_num][1].mean()
            imgs[img_num][2][y_min:y_max, x_min:x_max] = imgs[img_num][2].mean()
        #     for x_draw in range(x_min,x_max):
        #         for chan in range(3):
        #             imgs[img_num][0][y_min - 1+chan][x_draw] = 255
        #             imgs[img_num][1][y_min - 1+chan][x_draw] = 0
        #             imgs[img_num][2][y_min - 1+chan][x_draw] = 0
        #             imgs[img_num][0][y_max - 1+chan][x_draw] = 255
        #             imgs[img_num][1][y_max - 1+chan][x_draw] = 0
        #             imgs[img_num][2][y_max - 1+chan][x_draw] = 0
        #     for y_draw in range(y_min,y_max):
        #         for chan in range(3):
        #             imgs[img_num][0][y_draw][x_min - 1+chan] = 255
        #             imgs[img_num][1][y_draw][x_min - 1+chan] = 0
        #             imgs[img_num][2][y_draw][x_min - 1+chan] = 0
        #             imgs[img_num][0][y_draw][x_max - 1+chan] = 255
        #             imgs[img_num][1][y_draw][x_max - 1+chan] = 0
        #             imgs[img_num][2][y_draw][x_max - 1+chan] = 0
    return imgs

def paste_label_semantic(imgs, ss_model, size, random_number=False):
    imgs_new = []
    for img in imgs:
        img = F.interpolate(img[None, ...], size=size, mode='bilinear')
        mean_pixel_value_0 = img[0][0].mean()
        mean_pixel_value_1 = img[0][1].mean()
        mean_pixel_value_2 = img[0][2].mean()
        mean_img = torch.stack(
            [torch.full(img[0,0, ...].shape, mean_pixel_value_0), torch.full(img[0,1, ...].shape, mean_pixel_value_1),
             torch.full(img[0,2, ...].shape, mean_pixel_value_2)], dim=0).to(img.device)
        output = ss_model(img)
        output = torch.argmax(output['out'].squeeze(), dim=0).detach()
        output[output != 15] = 0
        output[output == 15] = 1
        paste_img = img * (1 - output[None,...]) + mean_img * output[None,...]
        imgs_new.append(paste_img)
    imgs_new = torch.stack(imgs_new, dim=0).squeeze(1)
    return imgs_new

def paste_label_value(imgs,real_labels, pseudo_label_value=-1): #-1时使用mean, -2时使用Random的值, 取正值时使用该值
    for di in range(len(real_labels)):
        img_num = int(real_labels[di][0])
        x_center = round(float(real_labels[di][2] * imgs.shape[3]))
        y_center = round(float(real_labels[di][3] * imgs.shape[2]))
        wide = round(float(real_labels[di][4] * imgs.shape[3]))
        height = round(float(real_labels[di][5] * imgs.shape[2]))
        x_min = round(max(x_center - wide / 2, 0))
        x_max = round(min(x_center + wide / 2, imgs.shape[3] - 1))
        y_min = round(max(y_center - height / 2, 0))
        y_max = round(min(y_center + height / 2, imgs.shape[2] - 1))
        if pseudo_label_value == -1:
            imgs[img_num][0][y_min:y_max, x_min:x_max] = imgs[img_num][0].mean()
            imgs[img_num][1][y_min:y_max, x_min:x_max] = imgs[img_num][1].mean()
            imgs[img_num][2][y_min:y_max, x_min:x_max] = imgs[img_num][2].mean()
        elif pseudo_label_value == -2:
            imgs[img_num][:][y_min:y_max, x_min:x_max] = torch.tensor(random.uniform(0,1)).to(imgs.device)
        else:
            imgs[img_num][0][y_min:y_max, x_min:x_max] = torch.tensor(pseudo_label_value).to(imgs.device)
            imgs[img_num][1][y_min:y_max, x_min:x_max] = torch.tensor(pseudo_label_value).to(imgs.device)
            imgs[img_num][2][y_min:y_max, x_min:x_max] = torch.tensor(pseudo_label_value).to(imgs.device)
    return imgs

def make_source_shortrange_weight_map(imgs, real_labels, weight_range=[1.,1.5,2.0,2.5,3.0],
                                      weight_add=0.25, foreground_value=1.):
    weight_map_list = torch.stack([torch.zeros(imgs.shape[0],imgs.shape[2],imgs.shape[3])] * len(weight_range), 0)
    weight_map_overall = torch.ones(imgs.shape[0],imgs.shape[2],imgs.shape[3])
    all_imgs_num = imgs.shape[0]
    for di in range(len(real_labels)):
        img_id = int(real_labels[di][0])

        x_center = round(float(real_labels[di][2] * imgs.shape[3]))
        y_center = round(float(real_labels[di][3] * imgs.shape[2]))
        wide = round(float(real_labels[di][4] * imgs.shape[3]))
        height = round(float(real_labels[di][5] * imgs.shape[2]))
        for range_i in range(len(weight_range)):
            weight_map_list[range_i][img_id][round(max(y_center - height * weight_range[range_i] / 2, 0)):
                                             round(min(y_center + height * weight_range[range_i] / 2, imgs.shape[2] - 1)),
                                             round(max(x_center - wide * weight_range[range_i] / 2, 0)):
                                             round(min(x_center + wide * weight_range[range_i] / 2, imgs.shape[3] - 1))] = 1
    for img_i in range(all_imgs_num):
        for map_list_i in range(len(weight_range)):
            weight_map_overall[img_i] += weight_map_list[map_list_i][img_i] * weight_add
        weight_map_overall[img_i][weight_map_list[0][img_i]==1] = foreground_value
    return weight_map_overall


def crop_front(imgs,real_labels):
    imgs_front = torch.zeros(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_front[i][0] = imgs[i][0].mean()
        imgs_front[i][1] = imgs[i][1].mean()
        imgs_front[i][2] = imgs[i][2].mean()
    for di in range(len(real_labels)):
        img_num = int(real_labels[di][0])
        x_center = round(float(real_labels[di][2] * imgs.shape[3]))
        y_center = round(float(real_labels[di][3] * imgs.shape[2]))
        wide = round(float(real_labels[di][4] * imgs.shape[3]))
        height = round(float(real_labels[di][5] * imgs.shape[2]))
        x_min = round(max(x_center - wide / 2, 0))
        x_max = round(min(x_center + wide / 2, imgs.shape[3] - 1))
        y_min = round(max(y_center - height / 2, 0))
        y_max = round(min(y_center + height / 2, imgs.shape[2] - 1))
        imgs_front[img_num][0][y_min:y_max, x_min:x_max] = imgs[img_num][0][y_min:y_max, x_min:x_max]
        imgs_front[img_num][1][y_min:y_max, x_min:x_max] = imgs[img_num][1][y_min:y_max, x_min:x_max]
        imgs_front[img_num][2][y_min:y_max, x_min:x_max] = imgs[img_num][2][y_min:y_max, x_min:x_max]
    return imgs_front


def paste_trg_preds(imgs=None,out_nms=None,all_black=False):
    for si, pred in enumerate(out_nms):
        if len(pred)==0:
            continue
        predn = pred.clone()
        box = xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2
        img_num = si
        for j in range(len(box)):
            x_min = round(float(box[j][0]))
            x_max = round(float(box[j][0])+float(box[j][2]))
            y_min = round(float(box[j][1]))
            y_max = round(float(box[j][1])+float(box[j][3]))
            if all_black:
                imgs[img_num][0][y_min:y_max, x_min:x_max] = 0
                imgs[img_num][1][y_min:y_max, x_min:x_max] = 0
                imgs[img_num][2][y_min:y_max, x_min:x_max] = 0
            else:
                imgs[img_num][0][y_min:y_max, x_min:x_max] = imgs[img_num][0].mean()
                imgs[img_num][1][y_min:y_max, x_min:x_max] = imgs[img_num][1].mean()
                imgs[img_num][2][y_min:y_max, x_min:x_max] = imgs[img_num][2].mean()
    return imgs

def paste_trg_preds_value(imgs=None, out_nms=None, pseudo_label_value=-1):
    for si, pred in enumerate(out_nms):
        if len(pred)==0:
            continue
        predn = pred.clone()
        box = xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2
        img_num = si
        for j in range(len(box)):
            x_min = round(float(box[j][0]))
            x_max = round(float(box[j][0])+float(box[j][2]))
            y_min = round(float(box[j][1]))
            y_max = round(float(box[j][1])+float(box[j][3]))
            if pseudo_label_value == -1:
                imgs[img_num][0][y_min:y_max, x_min:x_max] = imgs[img_num][0].mean()
                imgs[img_num][1][y_min:y_max, x_min:x_max] = imgs[img_num][1].mean()
                imgs[img_num][2][y_min:y_max, x_min:x_max] = imgs[img_num][2].mean()
            elif pseudo_label_value == -2:
                imgs[img_num][:][y_min:y_max, x_min:x_max] = random.uniform(0, 1)
            else:
                imgs[img_num][0][y_min:y_max, x_min:x_max] = pseudo_label_value
                imgs[img_num][1][y_min:y_max, x_min:x_max] = pseudo_label_value
                imgs[img_num][2][y_min:y_max, x_min:x_max] = pseudo_label_value
    return imgs

def make_target_shortrange_weight_map(imgs=None, out_nms=None, weight_range=[1.,1.5,2.0,2.5,3.0],
                                      weight_add=0.25, foreground_value=1.):
    weight_map_list = torch.stack([torch.zeros(imgs.shape[0], imgs.shape[2], imgs.shape[3])] * len(weight_range), 0)
    weight_map_overall = torch.ones(imgs.shape[0], imgs.shape[2], imgs.shape[3])
    all_imgs_num = imgs.shape[0]
    for si, pred in enumerate(out_nms):
        if len(pred)==0:
            continue
        predn = pred.clone()
        box = xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2
        img_id = si
        for j in range(len(box)):
            x_center = round(float(box[j][0]) + float(box[j][2]) / 2)
            wide = round(float(box[j][2]))
            y_center = round(float(box[j][1]) + float(box[j][3]) / 2)
            height = round(float(box[j][3]))

            for range_i in range(len(weight_range)):
                weight_map_list[range_i][img_id][round(max(y_center - height * weight_range[range_i] / 2, 0)):
                                                 round(min(y_center + height * weight_range[range_i] / 2, imgs.shape[2] - 1)),
                                                 round(max(x_center - wide * weight_range[range_i] / 2, 0)):
                                                 round(min(x_center + wide * weight_range[range_i] / 2, imgs.shape[3] - 1))] = 1
    for img_i in range(all_imgs_num):
        for map_list_i in range(len(weight_range)):
            weight_map_overall[img_i] += weight_map_list[map_list_i][img_i] * weight_add
        weight_map_overall[img_i][weight_map_list[0][img_i] == 1] = foreground_value
    return weight_map_overall


def out_trg_preds(imgs=None,out_nms=None):
    imgs_front = torch.zeros(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_front[i][0] = imgs[i][0].mean()
        imgs_front[i][1] = imgs[i][1].mean()
        imgs_front[i][2] = imgs[i][2].mean()
    for si, pred in enumerate(out_nms):
        if len(pred)==0:
            continue
        predn = pred.clone()
        box = xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2
        img_num = si
        for j in range(len(box)):
            x_min = round(float(box[j][0]))
            x_max = round(float(box[j][0])+float(box[j][2]))
            y_min = round(float(box[j][1]))
            y_max = round(float(box[j][1])+float(box[j][3]))
            imgs_front[img_num][0][y_min:y_max, x_min:x_max] = imgs[img_num][0][y_min:y_max, x_min:x_max]
            imgs_front[img_num][1][y_min:y_max, x_min:x_max] = imgs[img_num][1][y_min:y_max, x_min:x_max]
            imgs_front[img_num][2][y_min:y_max, x_min:x_max] = imgs[img_num][2][y_min:y_max, x_min:x_max]
    return imgs_front

def crop_front_list(imgs,real_labels,imgsz):#imgs:[b,3,2048,2048],real_labels:[di,6],imgsz=2048
    labels = []
    for di in range(len(real_labels)):
        img_num = int(real_labels[di][0])
        x_center = round(float(real_labels[di][2] * imgsz))
        y_center = round(float(real_labels[di][3] * imgsz))
        wide = round(float(real_labels[di][4] * imgsz))
        height = round(float(real_labels[di][5] * imgsz))
        x_min = round(max(x_center - wide / 2, 0))
        x_max = round(min(x_center + wide / 2, imgsz - 1))
        y_min = round(max(y_center - height / 2, 0))
        y_max = round(min(y_center + height / 2, imgsz - 1))
        # box = (x_min,y_min,x_max,y_max)
        new_label = torch.cat((imgs[img_num][0][y_min:y_max, x_min:x_max].unsqueeze(0),imgs[img_num][1][y_min:y_max, x_min:x_max].unsqueeze(0),imgs[img_num][2][y_min:y_max, x_min:x_max].unsqueeze(0)),0)
        labels.append(new_label.unsqueeze(0))
    return labels

def find_pict(pict_name,paths):
    i_num = -1
    if pict_name in paths:
        for i in range(len(paths)):
            if paths[i] == pict_name:
                i_num = i
                break
    return i_num

def deal_grad(model, key):
    for param in model.parameters():
        param.requires_grad = key
    return model

def cutf(feature, H=1.3):
    f_max = float(feature.max())
    f_mean = float(feature.mean())
    f_mean_full = torch.full(feature.shape,f_mean).to(feature.get_device())
    feature_new = torch.where(feature-f_mean>(f_max-f_mean)/H,f_mean_full,feature)
    return feature_new


def compute_loss_variance(feature,control_height=False,Layer=4):
    batch_num = len(feature[0])
    loss = 0
    for i in range(batch_num):
        feature_first = feature[0][i]
        feature_second = feature[1][i]
        feature_third = feature[2][i]
        feature_forth = feature[3][i]

        feature1 = torch.sum(feature_first, dim=0)
        feature2 = torch.sum(feature_second, dim=0)
        feature3 = torch.sum(feature_third, dim=0)
        feature4 = torch.sum(feature_forth, dim=0)

        f1_mean = feature1.mean()
        f2_mean = feature2.mean()
        f3_mean = feature3.mean()
        f4_mean = feature4.mean()
        if control_height:
            f1_max = feature1.max()
            f2_max = feature2.max()
            f3_max = feature3.max()
            f4_max = feature4.max()

            feature1 = cutf(feature1, H=1.5)
            feature2 = cutf(feature2, H=1.5)
            feature3 = cutf(feature3, H=1.5)
            feature4 = cutf(feature4, H=1.5)
        loss_1 = (ls_loss(feature1, f1_mean, reduction="mean"))**4
        loss_2 = (ls_loss(feature2, f2_mean, reduction="mean"))**4
        loss_3 = (ls_loss(feature3, f3_mean, reduction="mean"))**4
        loss_4 = (ls_loss(feature4, f4_mean, reduction="mean"))**4

        if Layer==4:
            loss += loss_1+loss_2+loss_3+loss_4
        if Layer==3:
            loss += loss_1 + loss_2 + loss_3
        if Layer==2:
            loss += loss_1 + loss_2
    return loss

def wetherCheat(epoch,MY_14=True):
    Cheat = False
    if MY_14 and epoch >= 1:#cyc改
        Cheat = True
    if not MY_14:
        Cheat = True
    return Cheat

def wetherCheat_help(epoch,MY_14=True):
    Cheat = False
    if MY_14 and epoch >= 3:#cyc改
        Cheat = True
    if not MY_14:
        Cheat = True
    return Cheat

def get_op_c(LEVEL=0):
    if LEVEL==1 or LEVEL==2:
        return [256,512,768,1024]
    if LEVEL==3:
        return [18,18,18,18]

def get_model(model):
    if model=='vit':
        return ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 2,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    channels=18,
    dropout = 0.1,
    emb_dropout = 0.1
)

    if model=='cvt':
        return CvT(
    num_classes = 2,
    s1_input_dim= 18,
    s1_emb_dim = 64,        # stage 1 - dimension
    s1_emb_kernel = 7,      # stage 1 - conv kernel
    s1_emb_stride = 4,      # stage 1 - conv stride
    s1_proj_kernel = 3,     # stage 1 - attention ds-conv kernel size
    s1_kv_proj_stride = 2,  # stage 1 - attention key / value projection stride
    s1_heads = 1,           # stage 1 - heads
    s1_depth = 1,           # stage 1 - depth
    s1_mlp_mult = 4,        # stage 1 - feedforward expansion factor
    s2_emb_dim = 192,       # stage 2 - (same as above)
    s2_emb_kernel = 3,
    s2_emb_stride = 2,
    s2_proj_kernel = 3,
    s2_kv_proj_stride = 2,
    s2_heads = 3,
    s2_depth = 2,
    s2_mlp_mult = 4,
    s3_emb_dim = 384,       # stage 3 - (same as above)
    s3_emb_kernel = 3,
    s3_emb_stride = 2,
    s3_proj_kernel = 3,
    s3_kv_proj_stride = 2,
    s3_heads = 4,
    s3_depth = 10,
    s3_mlp_mult = 4,
    dropout = 0.,
)
    if model == 'second':
        return CvT(
            num_classes=2,
            s1_input_dim=18,
            s1_emb_dim=64,  # stage 1 - dimension
            s1_emb_kernel=7,  # stage 1 - conv kernel
            s1_emb_stride=2,  # stage 1 - conv stride
            s1_proj_kernel=3,  # stage 1 - attention ds-conv kernel size
            s1_kv_proj_stride=2,  # stage 1 - attention key / value projection stride
            s1_heads=1,  # stage 1 - heads
            s1_depth=1,  # stage 1 - depth
            s1_mlp_mult=4,  # stage 1 - feedforward expansion factor
            s2_emb_dim=192,  # stage 2 - (same as above)
            s2_emb_kernel=3,
            s2_emb_stride=2,
            s2_proj_kernel=3,
            s2_kv_proj_stride=2,
            s2_heads=3,
            s2_depth=2,
            s2_mlp_mult=4,
            s3_emb_dim=384,  # stage 3 - (same as above)
            s3_emb_kernel=3,
            s3_emb_stride=2,
            s3_proj_kernel=3,
            s3_kv_proj_stride=2,
            s3_heads=4,
            s3_depth=10,
            s3_mlp_mult=4,
            dropout=0.,
        )

    if model == 'cvt_official':
        pth = 'CvT-13-224x224-IN-1k.pth'
        md = cvt_official.Base_cvt_cifar(pth)
        return md
    if model == 'cvt_simple':
        pth = 'CvT-13-224x224-IN-1k.pth'
        md = cvt_simple.Base_cvt_cifar(pth,stage_t=[1,1,1])
        return md
    if model == 'cvt_simple_18':
        pth = 'CvT-13-224x224-IN-1k.pth'
        md = cvt_simple_18.Base_cvt_cifar(pth,stage_t=[1,1,1])
        return md
    if model == 'cvt_hr':
        pth = 'CvT-13-224x224-IN-1k.pth'
        md = cvt_hr.Base_cvt_cifar(pth, stage_t=[1, 1, 3])
        return md
    if model == 'cvt_hr_18':
        pth = 'CvT-13-224x224-IN-1k.pth'
        md = cvt_hr_18.Base_cvt_cifar(pth, stage_t=[1, 1, 3])
        return md
    if model == 'cvt_hr_cat':
        pth = 'CvT-13-224x224-IN-1k.pth'
        md = cvt_hr_cat.Base_cvt_cifar(pth, stage_t=[1, 1, 3])
        return md
    if model == 'cvt_hr_bce':
        pth = 'CvT-13-224x224-IN-1k.pth'
        md = cvt_hr_bce.Base_cvt_cifar(pth, stage_t=[1, 1, 3])
        return md
    if model == 'cvt_official_18':
        pth = 'CvT-13-224x224-IN-1k.pth'
        md = cvt_official_18.Base_cvt_cifar(pth)
        return md

def get_front_or_back(pred):
    back_probability = []
    for i in range(len(pred)):
        pred[i] = pred[i].detach()
        prob = pred[i][:,:,:,:,4]
        prob_sum = pred[i].sum(4)
        back_probability.append(prob)
    return back_probability

def get_trg_mask(trg,level,NMS,C,I,D,size=(55,55)):   #trg:[3,20400,6]
    with amp.autocast(enabled=C):
        back = torch.ones(I.shape[0],1,I.shape[2],I.shape[3])  #[3,1,512,640]
        for i in range(len(level)):
            out_trg_NMS = non_max_suppression(trg, conf_thres=level[i], iou_thres=0.6, labels=[],
                                              multi_label=True)
            mask = (1-(paste_trg_preds(imgs=torch.ones(I.shape), out_nms=out_trg_NMS, all_black=True)[:, 0]).unsqueeze(1)) * level[i]
            #mask = paste_trg_preds_radar(imgs = torch.ones(back.shape), out_nms=out_trg_NMS, radar_size=[1.1,1.2,1.3,1.5,1.7,2.0,3.0,4.0,5.0])
            if i==0:
                mask_tensor = mask
            else:
                mask_tensor = torch.cat((mask_tensor,mask),1)
        all_mask = torch.max(mask_tensor, 1, keepdim=True)[0]
        if D:
            all_mask = all_mask * 10
        all_mask = torch.where(all_mask>=1,torch.ones(all_mask.shape),all_mask)
        back = back - all_mask
        #back = F.interpolate(back, size=(int(I.shape[2] / 8), int(I.shape[3] / 8)), mode='bilinear', align_corners=True)
        back = F.interpolate(back, size=size, mode='bilinear', align_corners=True)
    return back


def get_attention_mask(prob_mask):
    line = rearrange(prob_mask, 'b c h w -> b (h w) c').squeeze(-1).unsqueeze(1)
    back_attention_matrix = torch.matmul(torch.transpose(line,-2,-1),line)
    return back_attention_matrix

def paste_label_radar(imgs,real_labels,radar_size,center=0):
    radar_size.sort(reverse = True)
    if len(real_labels) == 0:
        return imgs
    for di in range(len(real_labels)):
        imgs_back = torch.ones(imgs.shape)
        img_num = int(real_labels[di][0])
        x_center = round(float(real_labels[di][2] * imgs.shape[3]))
        y_center = round(float(real_labels[di][3] * imgs.shape[2]))
        wide = float(real_labels[di][4] * imgs.shape[3])
        height = float(real_labels[di][5] * imgs.shape[2])

        for i in radar_size:
            x_min_r = round(max(x_center - wide * i / 2, 0))
            x_max_r = round(min(x_center + wide * i / 2, imgs.shape[3] - 1))
            y_min_r = round(max(y_center - height * i / 2, 0))
            y_max_r = round(min(y_center + height * i / 2, imgs.shape[2] - 1))
            imgs_back[img_num][0][y_min_r:y_max_r, x_min_r:x_max_r] = 1 + 1 / i
        if di == 0:
            imgs_back_tensor = imgs_back
        else:
            imgs_back_tensor = torch.cat((imgs_back_tensor,imgs_back),1)
    imgs = torch.max(imgs_back_tensor, 1, keepdim=True)[0]

    for di in range(len(real_labels)):
        img_num = int(real_labels[di][0])
        x_center = round(float(real_labels[di][2] * imgs.shape[3]))
        y_center = round(float(real_labels[di][3] * imgs.shape[2]))
        wide = round(float(real_labels[di][4] * imgs.shape[3]))
        height = round(float(real_labels[di][5] * imgs.shape[2]))
        x_min = round(max(x_center - wide / 2, 0))
        x_max = round(min(x_center + wide / 2, imgs.shape[3] - 1))
        y_min = round(max(y_center - height / 2, 0))
        y_max = round(min(y_center + height / 2, imgs.shape[2] - 1))
        imgs[img_num][0][y_min:y_max, x_min:x_max] = center
    return imgs

def paste_trg_preds_radar(trg,imgs,radar_size,conf_thres=0.01,center=0):
    W = 0
    all_len = 0
    radar_size.sort(reverse = True)
    out_trg_NMS = non_max_suppression(trg, conf_thres=conf_thres, iou_thres=0.6, labels=[],
                                      multi_label=True)
    for I in range(len(out_trg_NMS)):
        all_len += len(out_trg_NMS[I])
    if all_len == 0:
        return imgs
    for si, pred in enumerate(out_trg_NMS):
        if len(pred)==0:
            continue
        predn = pred.clone()
        box = xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2 #[7,4]
        img_num = si
        for j in range(len(box)):
            wide = float(box[j][2])  # 19.5
            height = float(box[j][3])
            x_center = round(float(box[j][0])+float(box[j][2])/2)
            y_center = round(float(box[j][1])+float(box[j][3])/2)
            imgs_back = torch.ones(imgs.shape) #[3,1,512,640]

            for i in radar_size:
                x_min_r = round(max(x_center - wide * i / 2, 0))
                x_max_r = round(min(x_center + wide * i / 2, imgs.shape[3] - 1))
                y_min_r = round(max(y_center - height * i / 2, 0))
                y_max_r = round(min(y_center + height * i / 2, imgs.shape[2] - 1))
                imgs_back[img_num][0][y_min_r:y_max_r, x_min_r:x_max_r] = 1 + 1 / i
            if W == 0:
                W = 1
                imgs_back_tensor = imgs_back
            else:
                imgs_back_tensor = torch.cat((imgs_back_tensor, imgs_back), 1)
    imgs = torch.max(imgs_back_tensor, 1, keepdim=True)[0]

    for si, pred in enumerate(out_trg_NMS):
        if len(pred)==0:
            continue
        predn = pred.clone()
        box = xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2
        img_num = si
        for j in range(len(box)):
            wide = float(box[j][2])  # 19.5
            height = float(box[j][3])
            x_center = round(float(box[j][0]) + float(box[j][2]) / 2)
            y_center = round(float(box[j][1]) + float(box[j][3]) / 2)
            x_min_r = round(max(x_center - wide / 2, 0))
            x_max_r = round(min(x_center + wide / 2, imgs.shape[3] - 1))
            y_min_r = round(max(y_center - height / 2, 0))
            y_max_r = round(min(y_center + height / 2, imgs.shape[2] - 1))
            imgs[img_num][0][y_min_r:y_max_r, x_min_r:x_max_r] = center

    return imgs

def change_C(feature):
    for i in range(len(feature)):
        if i == 0:
            feature_new = feature[i]
        else:
            feature_bilinear = F.interpolate(feature[i], size=(feature[0].shape[-2],feature[0].shape[-1]), mode='bilinear', align_corners=True)
            feature_new = torch.cat((feature_new,feature_bilinear),1)
    return feature_new

# if __name__ == '__main__':
#     with open()










