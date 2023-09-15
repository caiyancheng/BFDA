import os
import shutil

source_dir = '/remote-home/share/Cityscapes/leftImg8bit_foggy/'
aim_dir = '/remote-home/share/Cityscapes/newfoggy_cyc/images/'
s_list1 = os.listdir(source_dir)
if not os.path.exists(aim_dir):
    os.mkdir(aim_dir)

level_0005 = 0
level_001 = 0
level_002 = 0
all_pict = 0

for i in s_list1:
    sub_dir = source_dir+i
    s_list2 = os.listdir(sub_dir)
    for j in s_list2:
        sub_sub_dir = sub_dir+'/'+j
        s_list3 = os.listdir(sub_sub_dir)
        for k in s_list3:
            all_pict+=1
            print(all_pict)
            img_whole_name = sub_sub_dir+'/'+k
            level = k.split('.')[-2].split('_')[-1]
            if level == '005':
                level_0005 += 1
            if level == '01':
                level_001 += 1
            if level == '02':
                level_002 += 1
            img_aim_dir1 = aim_dir + '0.'+str(level)
            if not os.path.exists(img_aim_dir1):
                os.mkdir(img_aim_dir1)
            img_aim_dir2 = aim_dir + '0.' + str(level) + '/' + i
            if not os.path.exists(img_aim_dir2):
                os.mkdir(img_aim_dir2)
            img_aim_dir3 = aim_dir + '0.' + str(level) + '/' + i + '/' + j
            if not os.path.exists(img_aim_dir3):
                os.mkdir(img_aim_dir3)
            img_aim_name = aim_dir + '0.'+str(level) + '/' + i + '/' + j +'/'+k
            shutil.copy(img_whole_name, img_aim_name)

print('0.005:',level_0005)
print('0.01:',level_001)
print('0.02:',level_002)
