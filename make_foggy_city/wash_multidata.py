import os
import torch

m = 'val'
W = 0

num = 0
wrong_num = 0

ann_path = '/remote-home/share/42/cyc19307140030/yolov5/data/bdd10k/labels/'+m

ann_list = os.listdir(ann_path)
for i in ann_list:
    W = 0
    if not i.endswith('.txt'):
        continue
    num += 1
    print(num)
    txt_name = ann_path + '/' + i
    with open(txt_name, 'r') as fp:
        data = fp.readlines()
        real_len = len(set(data))
        if real_len != len(data):
            wrong_num += 1
            real_data = list(set(data))
            W = 1
    if W == 1:
        with open(txt_name, 'w') as fp:
            for a in real_data:
                fp.write(a)
print(wrong_num)

