import os
import json

m = 'train'
# bbx_path= '/remote-home/share/BDD100K/bdd100k_det_20_labels_trainval/bdd100k/labels/det_20/det_' + m + '.json'
label_path = '/remote-home/share/BDD100K/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_'+m+'.json'
pict_path = '/remote-home/share/BDD100K/bdd100k_images/bdd100k/images/100k/' + m + '/'
# aim_pict_path =
# aim_lb_path =


with open(label_path,'r') as fp:
    data = json.load(fp)

for i in data:
    W = 0#看有没有必要复制（只复制白天、有行人（pedestrian）的场景）
    pict_name = i['name']
    at = i['attributes']
    pict_label = i['labels']
    # for j in pict_label:

    real_pict_path = pict_path + pict_name

