import os
import shutil
import json

old_path = '/remote-home/share/42/cyc19307140030/yolov5/val_foggy_city.json'
aim_path_1 = '/remote-home/share/42/cyc19307140030/yolov5/val_foggy_city_0_096.json'
aim_path_2 = '/remote-home/share/42/cyc19307140030/yolov5/val_foggy_city_096_099.json'
aim_path_3 = '/remote-home/share/42/cyc19307140030/yolov5/val_foggy_city_099_1.json'
with open(old_path,'r') as fp:
    data = json.load(fp)