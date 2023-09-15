import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from torchvision import *
import cv2
from torch.nn import functional as F
import numpy as np
import os

def toTensor(img):
  assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = torch.from_numpy(img.transpose((2, 0, 1)))
  return img.float().div(255).unsqueeze(0) # 255也可以改为256

def tensor_to_np(tensor):
  img = tensor.mul(255).byte()
  img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
  return img

unloader = transforms.ToPILImage()

def show_feature(feature=None,img_id = 0,img_size = 0,save_dir = None,epoch = 0,remark=''):#只展示一个
    unloader = transforms.ToPILImage()
    if feature:
        if len(feature[0])>1:#说明传进来的信息的batch_size不是1
            feature = [feature[0][img_id].unsqueeze(0),feature[1][img_id].unsqueeze(0),feature[2][img_id].unsqueeze(0),feature[3][img_id].unsqueeze(0)]
        if img_size == 640:
            k = (480, 640)
            XX = 640
            YY = 480
            W = 8
        if img_size == 2048:
            k = (1024, 2048)
            XX = 2048
            YY = 1024
            W = 16
        A = False
        feature_1 = torch.sum(feature[0], dim=1, keepdim=True)
        feature_1 = F.interpolate(feature_1, k, mode='bilinear', align_corners=A)
        feature_show_1 = unloader(feature_1.squeeze(0))

        feature_2 = torch.sum(feature[1], dim=1,keepdim=True)
        feature_2 = F.interpolate(feature_2, k, mode='bilinear', align_corners=A)
        feature_show_2 = unloader(feature_2.squeeze(0))

        feature_3 = torch.sum(feature[2], dim=1, keepdim=True)
        feature_3 = F.interpolate(feature_3, k, mode='bilinear', align_corners=A)
        feature_show_3 = unloader(feature_3.squeeze(0))

        feature_4 = torch.sum(feature[3], dim=1, keepdim=True)
        feature_4 = F.interpolate(feature_4, k, mode='bilinear', align_corners=A)
        feature_show_4 = unloader(feature_4.squeeze(0))

        for i in range(4):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            X = np.arange(0, XX, W)  # (0,640,4)
            Y = np.arange(0, YY, W)  # (0,480,4)
            X, Y = np.meshgrid(X, Y)
            if i == 0:
                feature_1 = F.interpolate(feature_1.clone(), (int(YY / W), int(XX / W)),mode='bilinear',align_corners=A)
                Z = feature_1.clone().detach().cpu().numpy()[0][0]
            if i == 1:
                feature_2 = F.interpolate(feature_2.clone(), (int(YY / W), int(XX / W)),mode='bilinear',align_corners=A)
                Z = feature_2.clone().detach().cpu().numpy()[0][0]
            if i == 2:
                feature_3 = F.interpolate(feature_3.clone(), (int(YY / W), int(XX / W)),mode='bilinear',align_corners=A)
                Z = feature_3.clone().detach().cpu().numpy()[0][0]
            if i == 3:
                feature_4 = F.interpolate(feature_4.clone(), (int(YY / W), int(XX / W)),mode='bilinear',align_corners=A)
                Z = feature_4.clone().detach().cpu().numpy()[0][0]
            # Z = Z - Z.min() #(更好看)
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

            # ax.set_zlim(Z.min(), Z.max())
            ax.set_zlim(-40, 10)
            ax.zaxis.set_major_locator(LinearLocator(10))  # 设置Z轴间隔
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            fig.colorbar(surf, shrink=0.5, aspect=5)

            save_dir_epoch = str(save_dir)+"/epoch"+str(epoch)
            if not os.path.exists(save_dir_epoch):
                os.mkdir(save_dir_epoch)
            save_dir_remark = save_dir_epoch+'/'+remark
            if not os.path.exists(save_dir_remark):
                os.mkdir(save_dir_remark)
            plt.savefig(str(save_dir_remark) +"/layer"+ str(i+1)+".png")
            del ax
            plt.close()


        # if not training:
        #     cv2.imwrite(str(save_dir) + '/feature_first_afterconv' + str(batch_i) + '.png',
        #                 np.array(feature_show_1_afterconv))
        #     cv2.imwrite(str(save_dir) + '/feature_second_afterconv' + str(batch_i) + '.png',
        #                 np.array(feature_show_2_afterconv))
        #     cv2.imwrite(str(save_dir) + '/feature_third_afterconv' + str(batch_i) + '.png',
        #                 np.array(feature_show_3_afterconv))
        #     cv2.imwrite(str(save_dir) + '/feature_forth_afterconv' + str(batch_i) + '.png',
        #                 np.array(feature_show_4_afterconv))

def show_feature_cam(feature_map,img,id,save_dir):
    feature_map = feature_map.float()
    feature_map = F.interpolate(feature_map, (img.shape[-2],img.shape[-1]), mode='bilinear', align_corners=True)
    feature_map = feature_map[id].unsqueeze(0)
    feature_map_show_sum = torch.sum(feature_map, dim=1,keepdim=True)
    feature_map_show_sum = (feature_map_show_sum-feature_map_show_sum.min())/(feature_map_show_sum.max()-feature_map_show_sum.min())
    # feature_map_show_max = feature_map.max(1)
    # feature_map_show_mean = feature_map.mean(1)
    feature_map_show_sum = tensor_to_np(feature_map_show_sum)
    heat_map = cv2.applyColorMap(feature_map_show_sum, cv2.COLORMAP_JET)
    # cv2.imwrite(save_dir+'heat_map.jpg',np.array(heat_map))
    # plt.figure()
    # plt.imshow(heat_map.squeeze(0))
    # plt.show()
    img = tensor_to_np(img)
    im_show = img * 0.6 + heat_map * 0.4
    cv2.imwrite(save_dir+'deep_cam.jpg', np.array(im_show))
    # plt.figure()
    # plt.imshow(im_show.squeeze(0))
    # plt.show()

def show_bbx(save_dir,path,bbx):
    path = str(path)
    img = cv2.imread(str(path))
    img_t = toTensor(img)
    for i in bbx:
        img_copy = img_t.clone()
        x_l, y_l, w, h = i
        x_l = float(x_l)
        y_l = float(y_l)
        w = float(w)
        h = float(h)
        X_L = round(x_l)
        W =  w
        Y_U = round(y_l)
        H = h
        X_R = round(X_L + W)
        Y_B = round(Y_U + H)
        K = 3
        img_t[0][0][Y_U - K:Y_B + K, X_L - K:X_R + K] = 1.
        img_t[0][1][Y_U - K:Y_B + K, X_L - K:X_R + K] = 0.
        img_t[0][2][Y_U - K:Y_B + K, X_L - K:X_R + K] = 0.
        img_t[0][0][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1] = img_copy[0][0][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1]
        img_t[0][1][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1] = img_copy[0][1][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1]
        img_t[0][2][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1] = img_copy[0][2][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1]

    # lb_path = '/remote-home/share/42/cyc19307140030/yolov5/data/' + path.split('/')[1] + '/labels/' + path.split('/')[
    #     3] + '/' + path.split('/')[4] + '/' + path.split('/')[-1].split('.')[0] + '.txt'
    lb_path = path.replace('png','txt')
    lb_path = lb_path.replace('images', 'labels')

    with open(lb_path, 'r') as fp:
        lb_data = fp.readlines()

    for i in lb_data:
        img_copy = img_t.clone()
        n, x_c, y_c, w, h = i.split('\t')
        x_c = float(x_c)
        y_c = float(y_c)
        w = float(w)
        h = float(h)
        X_c = img_t.shape[3] * x_c
        W = img_t.shape[3] * w
        Y_c = img_t.shape[2] * y_c
        H = img_t.shape[2] * h
        X_L = round(X_c - W / 2)
        X_R = round(X_c + W / 2)
        Y_U = round(Y_c - H / 2)
        Y_B = round(Y_c + H / 2)
        K = 3
        img_t[0][0][Y_U - K:Y_B + K, X_L - K:X_R + K] = 0.
        img_t[0][1][Y_U - K:Y_B + K, X_L - K:X_R + K] = 1.
        img_t[0][2][Y_U - K:Y_B + K, X_L - K:X_R + K] = 0
        img_t[0][0][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1] = img_copy[0][0][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1]
        img_t[0][1][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1] = img_copy[0][1][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1]
        img_t[0][2][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1] = img_copy[0][2][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1]
    img_show = unloader(img_t[0])
    sub_dir = '/remote-home/share/42/cyc19307140030/CVPR/BFNet/Caltech/ours/bbx/' + \
              path.split('/')[-1].split('.')[0].split('_')[0]
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    aim_path = sub_dir + '/' + path.split('/')[-1].split('.')[0] + "_bbx" + ".png"
    cv2.imwrite(aim_path, np.array(img_show)[..., ::-1])

def show_bbx_foggy(save_dir,path,bbx):
    path = str(path)
    img = cv2.imread(str(path))
    img_t = toTensor(img)
    for i in bbx:
        img_copy = img_t.clone()
        x_l, y_l, w, h = i
        x_l = float(x_l)
        y_l = float(y_l)
        w = float(w)
        h = float(h)
        X_L = round(x_l)
        W =  w
        Y_U = round(y_l)
        H = h
        X_R = round(X_L + W)
        Y_B = round(Y_U + H)
        K = 8
        img_t[0][0][Y_U - K:Y_B + K, X_L - K:X_R + K] = 1.
        img_t[0][1][Y_U - K:Y_B + K, X_L - K:X_R + K] = 0.
        img_t[0][2][Y_U - K:Y_B + K, X_L - K:X_R + K] = 0.
        img_t[0][0][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1] = img_copy[0][0][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1]
        img_t[0][1][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1] = img_copy[0][1][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1]
        img_t[0][2][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1] = img_copy[0][2][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1]

    # lb_path = '/remote-home/share/42/cyc19307140030/yolov5/data/' + path.split('/')[1] + '/labels/' + path.split('/')[
    #     3] + '/' + path.split('/')[4] + '/' + path.split('/')[-1].split('.')[0] + '.txt'
    lb_path = path.replace('png','txt')
    lb_path = lb_path.replace('images', 'labels')

    with open(lb_path, 'r') as fp:
        lb_data = fp.readlines()

    for i in lb_data:
        img_copy = img_t.clone()
        n, x_c, y_c, w, h = i.split('\t')
        x_c = float(x_c)
        y_c = float(y_c)
        w = float(w)
        h = float(h)
        X_c = img_t.shape[3] * x_c
        W = img_t.shape[3] * w
        Y_c = img_t.shape[2] * y_c
        H = img_t.shape[2] * h
        X_L = round(X_c - W / 2)
        X_R = round(X_c + W / 2)
        Y_U = round(Y_c - H / 2)
        Y_B = round(Y_c + H / 2)
        K = 8
        img_t[0][0][Y_U - K:Y_B + K, X_L - K:X_R + K] = 0.
        img_t[0][1][Y_U - K:Y_B + K, X_L - K:X_R + K] = 1.
        img_t[0][2][Y_U - K:Y_B + K, X_L - K:X_R + K] = 0
        img_t[0][0][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1] = img_copy[0][0][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1]
        img_t[0][1][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1] = img_copy[0][1][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1]
        img_t[0][2][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1] = img_copy[0][2][Y_U + 1:Y_B - 1, X_L + 1:X_R - 1]
    img_show = unloader(img_t[0])
    sub_dir = '/remote-home/share/42/cyc19307140030/CVPR/BFNet/FoggyCityscapes/ours/' + \
              path.split('/')[-1].split('.')[0].split('_')[0]
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    aim_path = sub_dir + '/' + path.split('/')[-1].split('.')[0] + "_bbx" + ".png"
    cv2.imwrite(aim_path, np.array(img_show)[..., ::-1])


if __name__ == '__main__':
    f = torch.randn(4,64,55,55)
    f = F.interpolate(f, (1024,2048), mode='bilinear', align_corners=True)
    img_path = '/remote-home/share/42/cyc19307140030/yolov5/runs/trainforpaper_crossdomain_transformer/exp37/src_img_aachen_000031_000019.png'
    img = cv2.imread(img_path)
    img_t = toTensor(img)
    img_show = unloader(img)
    cv2.imwrite('/remote-home/share/42/cyc19307140030/yolov5/debug/show/cam/source_pict.jpg', np.array(img_show))
    i = 0
    show_feature_cam(feature_map=f, img=img_t, id=i)