import os
import shutil

M = 'val'

num = 0

img_thick_path = '/remote-home/share/DAOD_Dataset/More_Dense_FoggyCityscape/thicker_foggy/'+M+'/JPEGImages/'

img_002_path = '/remote-home/share/Cityscapes/newfoggy_cyc/images/0.02/'+M+'/'

img_001_path = '/remote-home/share/Cityscapes/newfoggy_cyc/images/0.01/'+M+'/'

img_0005_path = '/remote-home/share/Cityscapes/newfoggy_cyc/images/0.005/'+M+'/'

lb_path_train = '/remote-home/share/Cityscapes/all_class/foggycityscapes/labels/'+M+'/'

aim_base_path = '/remote-home/share/Cityscapes/newfoggy_cyc/images/all_1_thickonly/'

aim_img_path_train = aim_base_path+M+'/'

aim_lb_path_train = aim_base_path.replace("images","labels")+M+'/'

if not os.path.exists(aim_base_path.replace("images","labels")):
    os.mkdir(aim_base_path.replace("images","labels"))
if not os.path.exists(aim_base_path):
    os.mkdir(aim_base_path)
if not os.path.exists(aim_img_path_train):
    os.mkdir(aim_img_path_train)
if not os.path.exists(aim_lb_path_train):
    os.mkdir(aim_lb_path_train)

#处理train
dir = os.listdir(img_002_path)
for i in dir:
    aim_dir = aim_img_path_train+i
    if not os.path.exists(aim_dir):
        os.mkdir(aim_dir)
    sub_dir = os.listdir(img_002_path+i)
    for j in sub_dir:
        num += 1
        print(num)
        img_name = j.split('.')[0]
        new_lb_name = j.split('.')[0]
        ###############转移img
        old_img_path_002 = img_002_path + i + '/' + img_name + '.02.png'
        old_img_path_001 = img_001_path + i + '/' + img_name + '.01.png'
        old_img_path_0005 = img_0005_path + i + '/' + img_name + '.005.png'
        old_img_path_thick = img_thick_path + 'target_' + img_name + '.02.jpg'

        new_img_path_002 = aim_dir + '/' + img_name + '02.png'
        new_img_path_001 = aim_dir + '/' + img_name + '01.png'
        new_img_path_0005 = aim_dir + '/' + img_name + '005.png'
        new_img_path_thick = aim_dir + '/' + img_name + 'thick.png'
        ###############转移lb
        old_lb_path = lb_path_train+ i + '/' + img_name + '.02.txt'
        aim_lb_dir = aim_lb_path_train+i
        if not os.path.exists(aim_lb_dir):
            os.mkdir(aim_lb_dir)
        new_lb_path_002 = aim_lb_dir + '/' + new_lb_name + '02.txt'
        new_lb_path_001 = aim_lb_dir + '/' + new_lb_name + '01.txt'
        new_lb_path_0005 = aim_lb_dir + '/' + new_lb_name + '005.txt'
        new_lb_path_thick = aim_lb_dir + '/' + new_lb_name + 'thick.txt'

        # #复制
        # shutil.copy(old_img_path_002,new_img_path_002)
        shutil.copy(old_img_path_thick,new_img_path_thick)
        # shutil.copy(old_img_path_001,new_img_path_001)
        # shutil.copy(old_img_path_0005,new_img_path_0005)

        # shutil.copy(old_lb_path,new_lb_path_002)
        # shutil.copy(old_lb_path,new_img_path_001)
        # shutil.copy(old_lb_path,new_img_path_0005)
        shutil.copy(old_lb_path,new_lb_path_thick)
