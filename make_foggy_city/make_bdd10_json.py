import json
import os

all_num = 0
n_a = 0

example_json_path = '/remote-home/share/42/cyc19307140030/yolov5/val_gt.json'
aim_json_path = '/remote-home/share/42/cyc19307140030/yolov5/val_bdd10.json'
lb_path = '/remote-home/share/BDD100k_onlyperson/BDD10k/labels/val/'
bdd_dict_path = '/remote-home/share/42/cyc19307140030/yolov5/bdd_10k_dict.json'
example_dict = '/remote-home/share/42/cyc19307140030/yolov5/foggy_city_dict.json'

with open(example_json_path,'r') as fp1: #示例json
    data_ex = json.load(fp1)
with open(example_dict,'r') as fp2: #示例dict
    data_dict = json.load(fp2)

data_aim_json = {'categories':[{'id':1,'name':'person'}],'images':[],'annotations':[]}
bdd_dict = {}

lb_dir = os.listdir(lb_path)
for i in lb_dir:
    txt_path = lb_path+i
    all_num += 1
    print(all_num)
    bdd_dict[i.split('.')[0]] = str(all_num)
    im_name = i.split('.')[0]+'.jpg'
    dict_i = {'id': all_num,'im_name': im_name ,'height': 720,'width': 1280}
    data_aim_json['images'].append(dict_i)
    with open(txt_path,'r') as fp3:
        data_lb = fp3.readlines()
    for j in data_lb:
        if i == '7e23ea75-4e570000.txt':
            hen = 1
        n_a += 1
        detial = j.split('\t')
        x_min = round((float(detial[1])-float(detial[3])/2)*1280)
        y_min = round((float(detial[2])-float(detial[4])/2)*720)
        w = round(float(detial[3])*1280)
        h = round(float(detial[4])*720)
        dict_a = {'id': n_a, 'image_id': all_num, 'category_id': 1, 'iscrowd': 0, 'ignore': 0,
                  'bbox': [x_min, y_min, w, h], 'vis_bbox': [x_min, y_min, w, h], 'height': h, 'vis_ratio': 1}
        data_aim_json['annotations'].append(dict_a)

        # # dict_a = {'id': n_a,'image_id': all_num, 'iscrowd': 0, 'ignore': 0, 'bbox': []}
                # x_min = j['box2d']['x1']
                # y_min = j['box2d']['y1']
                # w = j['box2d']['x2'] - j['box2d']['x1']
                # h = j['box2d']['y2'] - j['box2d']['y1']
                # dict_a = {'id': n_a, 'image_id': all_num, 'category_id': 1, 'iscrowd': 0, 'ignore': 0, 'bbox': [x_min, y_min, w, h], 'vis_bbox': [x_min, y_min, w, h], 'height': h, 'vis_ratio': 1}
                # data_aim_json['annotations'].append(dict_a)

with open(aim_json_path,'w') as fp4:
    json.dump(data_aim_json,fp4)

with open(bdd_dict_path, 'w') as fp5:
    json.dump(bdd_dict,fp5)
print(all_num)
