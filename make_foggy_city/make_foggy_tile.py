import os
import shutil

m = 'train'
all_pict = 0

source_dir ='/remote-home/share/Cityscapes/citypersons/'+m+'_full/'
aim_dir = '/remote-home/share/42/cyc19307140030/yolov5/data/foggycityscapes/labels/'+m+'_0.02/'

#munster_000173_000019_leftImg8bit_foggy_beta_0.02.png
#munster_000173_000019_leftImg8bit.txt
source_list1 = os.listdir(source_dir)

for i in source_list1:
    if i.endswith('.cache'):
        continue
    if i.endswith('_full'):
        continue
    sub_dir = source_dir+i
    source_list2 = os.listdir(sub_dir)
    for j in source_list2:
        all_pict+=1
        print(all_pict)
        newtxt = j.split('.')[0]+'_foggy_beta_0.02.txt'
        aim_path1 = aim_dir+i
        if not os.path.exists(aim_path1):
            os.mkdir(aim_path1)
        real_aim = aim_path1+'/'+newtxt
        old_path = source_dir+i+'/'+j
        shutil.copy(old_path,real_aim)

