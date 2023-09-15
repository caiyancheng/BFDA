import os
import json

old_txt_path = '/remote-home/share/42/cyc19307140030/yolov5/city_val_dict_reverse.json'
new_dict_path = '/remote-home/share/42/cyc19307140030/yolov5/foggy_city_dict.json'

with open(old_txt_path,'r') as fp:
    data = json.load(fp)

new_dict = {}

for i in data.items():
    new_key = i[0]+ '_foggy_beta_0.02'
    new_value = i[1]
    new_dict[new_key] = new_value

with open(new_dict_path,'w') as f:
    json.dump(new_dict,f)



