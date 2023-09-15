import json
import os

all_num = 0
n_a = 0

example_json_path = '/remote-home/share/42/cyc19307140030/yolov5/val_gt.json'
aim_json_path = '/remote-home/share/42/cyc19307140030/yolov5/val_bdd100_night.json'
lb_path = '/remote-home/share/BDD100K/bdd100k_det_20_labels_trainval/bdd100k/labels/det_20/det_val.json'
bdd_dict_path = '/remote-home/share/42/cyc19307140030/yolov5/bdd_night_dict.json'
example_dict = '/remote-home/share/42/cyc19307140030/yolov5/foggy_city_dict.json'

with open(example_json_path,'r') as fp:
    data_ex = json.load(fp)
with open(lb_path,'r') as fpp:
    data_lb = json.load(fpp)
with open(example_dict,'r') as fpppp:
    data_dict = json.load(fpppp)

data_aim_json = {'categories':[{'id':1,'name':'pedestrian'}],'images':[],'annotations':[]}
bdd_dict = {}

for i in data_lb:
    W = 0
    if i['attributes']['timeofday'] != 'night':
        continue
    real_lb = i['labels']
    for j in real_lb:
        if j['category'] == 'pedestrian':
            W = 1
    if W == 1:
        all_num += 1
        bdd_dict[i['name'].split('.')[0]] = str(all_num)
        dict_i = {'id': all_num,'im_name': i['name'],'height': 720,'width': 1280}
        data_aim_json['images'].append(dict_i)
        for j in real_lb:
            if j['category'] == 'pedestrian':
                n_a += 1
                # dict_a = {'id': n_a,'image_id': all_num, 'iscrowd': 0, 'ignore': 0, 'bbox': []}
                x_min = j['box2d']['x1']
                y_min = j['box2d']['y1']
                w = j['box2d']['x2'] - j['box2d']['x1']
                h = j['box2d']['y2'] - j['box2d']['y1']
                dict_a = {'id': n_a, 'image_id': all_num, 'category_id': 1, 'iscrowd': 0, 'ignore': 0, 'bbox': [x_min, y_min, w, h], 'vis_bbox': [x_min, y_min, w, h], 'height': h, 'vis_ratio': 1}
                data_aim_json['annotations'].append(dict_a)

with open(aim_json_path,'w') as fppp:
    json.dump(data_aim_json,fppp)

with open(bdd_dict_path, 'w') as fppppp:
    json.dump(bdd_dict,fppppp)
print(all_num)
