import os
import shutil
import json

old_path = '/remote-home/share/42/cyc19307140030/yolov5/val_gt.json'
aim_path = '/remote-home/share/42/cyc19307140030/yolov5/val_foggy_city.json'

with open(old_path,'r') as fp:
    data = json.load(fp)

for i in data['images']:
    i['im_name'] = i['im_name'].split('.')[0]+'_foggy_beta_0.02.png'

with open(aim_path,'w') as f:
    json.dump(data,f)