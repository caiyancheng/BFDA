import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm
from matplotlib.figure import Figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cbook import get_sample_data
import cv2
from matplotlib._png import read_png
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized

from MR_2 import validate
from torchvision import *
import cv2
from torch.nn import functional as F
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


unloader = transforms.ToPILImage()

def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,#640
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         is_coco=False,
         MR=True,
         Datasets = 'citytocaltech',  #'citytocaltech',#'foggycity'
         vision_show_feature = False,
         iter = 0, epoch = 0, in_iter = False, img_size = 640,
         ):#use MR-2 instead of MAP
    # Initialize/load model and set device
    if Datasets == 'cityperson' or Datasets == 'caltechtocity':
        dict_path = 'city_val_dict_reverse.json'
    if Datasets=='foggycity' or Datasets == 'citytofoggy':
        dict_path = 'foggy_city_dict.json'
    if Datasets == 'caltech' or Datasets == 'citytocaltech':
        dict_path = 'caltech_dict.json'
    if Datasets == 'bdd_day' or Datasets == 'citytobdd_day':
        dict_path = 'bdd_day_dict.json'
    if Datasets == 'bdd_night' or Datasets == 'citytobdd_night':
        dict_path = 'bdd_night_dict.json'
    if Datasets == 'bdd10k':
        dict_path = 'bdd_10k_dict.json'

    with open(dict_path, 'r', encoding='utf8') as fp1:
        img_dict = json.load(fp1)

    training = model is not None#False
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run#runs/test/exp14
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()#10

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0#初始化测试的图片数量
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    #s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')#cyc
    #if MR:
        #s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'MR_2', 'mAP@.5', 'mAP@.5:.95')
    #else:
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []#初始化json文件的字典
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        W = 0
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, train_out, feature, feature_afterconv, feature_after_backbone = model(img, augment=augment)  # inference and training outputs#, vision_feature=True
            # else:
            #     out, train_out = model(img, augment=augment)
            t0 += time_synchronized() - t
            # for NUM_i in range(len(paths)):
            #     if paths[NUM_i].split('/')[-1] == 'frankfurt_000000_001016_leftImg8bit.png':
            #         W = 1
            #         print("find it")
            #         continue
            if vision_show_feature:#paths[0].split('/')[-1] == 'frankfurt_000001_013016_leftImg8bit.png': #and paths[0].split('/')[-1] == 'set07_V000_179.png':
                only_first = True
                # feature_first = torch.max(feature[0],dim=1)[0]
                # feature_first = 255*torch.sum(feature[0], dim=1, keepdim=True)/feature[0].max()
                # feature_first = F.interpolate(feature_first, (1024, 2048), mode='bilinear',
                #                                      align_corners=True)
                # feature_show_first = unloader(feature_first.squeeze(0))
                # # feature_second = torch.max(feature[1], dim=1)[0]
                # feature_second = 255*torch.sum(feature[1], dim=1, keepdim=True)/feature[1].max()
                # feature_second = F.interpolate(feature_second, (1024, 2048), mode='bilinear',
                #                                       align_corners=True)
                # feature_show_second = unloader(feature_second.squeeze(0))
                # # feature_third = torch.max(feature[2], dim=1)[0]
                # feature_third = 255*torch.sum(feature[2], dim=1, keepdim=True)/feature[2].max()
                # feature_third = F.interpolate(feature_third, (1024, 2048), mode='bilinear',
                #                                      align_corners=True)
                # feature_show_third = unloader(feature_third.squeeze(0))
                # # feature_forth = torch.max(feature[3], dim=1)[0]
                # feature_forth = 255*torch.sum(feature[3], dim=1, keepdim=True)/feature[3].max()
                # feature_forth = F.interpolate(feature_forth, (1024, 2048), mode='bilinear',
                #                                      align_corners=True)
                # feature_show_forth = unloader(feature_forth.squeeze(0))
                # cv2.imwrite(str(save_dir) + '/feature_first'+str(batch_i)+'.png', np.array(feature_show_first))
                # cv2.imwrite(str(save_dir) + '/feature_second'+str(batch_i)+'.png', np.array(feature_show_second))
                # cv2.imwrite(str(save_dir) + '/feature_third'+str(batch_i)+'.png', np.array(feature_show_third))
                # cv2.imwrite(str(save_dir) + '/feature_forth'+str(batch_i)+'.png', np.array(feature_show_forth))
                #############################################after_conv
                if img_size == 640:
                    k = (480,640)
                if img_size == 2048:
                    k = (1024,2048)
                A = False
                NUM_i = 0
                feature_1_afterconv = torch.sum(feature_afterconv[0][NUM_i].unsqueeze(0), dim=1, keepdim=True)# / 5.#feature_afterconv[0].max() *255.
                feature_1_afterconv = F.interpolate(feature_1_afterconv, k, mode='bilinear',#(1024,2048)(480, 640)
                                              align_corners=A)
                feature_show_1_afterconv = unloader(feature_1_afterconv.squeeze(0))
                # feature_2 = torch.max(feature[1], dim=1)[0]
                if not only_first:
                    feature_2_afterconv = torch.sum(feature_afterconv[1][NUM_i].unsqueeze(0), dim=1,
                                                    keepdim=True)  # / 5.#feature_afterconv[1].max() *255.
                    feature_2_afterconv = F.interpolate(feature_2_afterconv, k, mode='bilinear',
                                                        align_corners=A)
                    feature_show_2_afterconv = unloader(feature_2_afterconv.squeeze(0))
                    # feature_third = torch.max(feature[2], dim=1)[0]
                    feature_3_afterconv = torch.sum(feature_afterconv[2][NUM_i].unsqueeze(0), dim=1,
                                                    keepdim=True)  # / 5.#feature_afterconv[2].max() *255.
                    feature_3_afterconv = F.interpolate(feature_3_afterconv, k, mode='bilinear',
                                                        align_corners=A)
                    feature_show_3_afterconv = unloader(feature_3_afterconv.squeeze(0))
                    # feature_forth = torch.max(feature[3], dim=1)[0]
                    feature_4_afterconv = torch.sum(feature_afterconv[3][NUM_i].unsqueeze(0), dim=1,
                                                    keepdim=True)  # / 5.#feature_afterconv[3].max() *255.
                    feature_4_afterconv = F.interpolate(feature_4_afterconv, k, mode='bilinear',
                                                        align_corners=A)
                    feature_show_4_afterconv = unloader(feature_4_afterconv.squeeze(0))


                # feature_for_mat_show = {feature_first_afterconv,feature_second_afterconv,feature_third_afterconv,feature_forth_afterconv}
                # mat_dir = str(save_dir) + '/mat_show.json'
                # with open(mat_dir,'w') as fp:
                #     json.dump(feature_for_mat_show,fp)
                ###feature_third_afterconv.clone().cpu().numpy()[0][0]
                if img_size == 640:
                    XX = 640
                    YY = 480
                    Ww = 2
                if img_size == 2048:
                    XX = 2048
                    YY = 1024
                    Ww = 8
                if only_first:
                    ir = 1
                else:
                    ir = 4
                for i in range(ir):
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    X = np.arange(0,XX,Ww)#(0,640,4)
                    Y = np.arange(0,YY,Ww)#(0,480,4)
                    X, Y = np.meshgrid(X, Y)
                    if i == 0:
                        feature_1_afterconv = F.interpolate(feature_1_afterconv.clone(), (int(YY/Ww), int(XX/Ww)), mode='bilinear',#(128, 256)
                                  align_corners=A)
                        Z = feature_1_afterconv.clone().cpu().numpy()[0][0]
                    if i == 1:
                        feature_2_afterconv = F.interpolate(feature_2_afterconv.clone(), (int(YY/Ww), int(XX/Ww)), mode='bilinear',#(120, 160)
                                  align_corners=A)
                        Z = feature_2_afterconv.clone().cpu().numpy()[0][0]
                    if i == 2:
                        feature_3_afterconv = F.interpolate(feature_3_afterconv.clone(), (int(YY/Ww), int(XX/Ww)), mode='bilinear',
                                  align_corners=A)
                        Z = feature_3_afterconv.clone().cpu().numpy()[0][0]
                    if i == 3:
                        feature_4_afterconv = F.interpolate(feature_4_afterconv.clone(), (int(YY/Ww), int(XX/Ww)), mode='bilinear',
                                  align_corners=A)
                        Z = feature_4_afterconv.clone().cpu().numpy()[0][0]
                    # Z = Z - Z.min() #(更好看)

                    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), alpha = 0.8)

                    ax.set_zlim(Z.min(), Z.max())
                    ax.zaxis.set_major_locator(LinearLocator(10))#设置Z轴间隔
                    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

                    img_c = cv2.imread("/remote-home/share/Caltech/images/test_show/"+paths[0].split('/')[-2]+'/'+paths[0].split('/')[-1])
                    img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
                    img_c = img_c.astype('float32') / 255
                    img_c = np.einsum('wlc->lwc', img_c)
                    # img = cv2.flip(img, 1)
                    img_c = cv2.flip(img_c, 1)
                    x, y = np.ogrid[0:img_c.shape[0], 0:img_c.shape[1]]
                    ax.plot_surface(x, y, np.atleast_2d(Z.min()), rstride=5, cstride=5, facecolors=img_c)


                    font = {'family': 'serif',
                            'color': 'red',
                            'weight': 'normal',
                            'size': 16,}

                    # ax.set_xlabel('length(L)',fontdict=font)
                    # ax.set_ylabel('width(W)',fontdict=font)
                    # ax.set_zlabel('feature value(V)',fontdict=font)
                    # ax.set_xlabel('L', fontdict=font)
                    # ax.set_ylabel('W', fontdict=font)
                    # ax.set_zlabel('V', fontdict=font)
                    # fig.colorbar(surf, shrink=0.5, aspect=5)

                    if not training:
                        plt.savefig(str(save_dir) + '/'+paths[0].split('/')[-1].split('.')[0] +"height_B" + str(batch_i) + "_" + str(i + 1) + ".png", bbox_inches='tight',dpi=200)
                        plt.close()


            # Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if MR:
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    new_id = img_dict[image_id]
                    jdict.append({'image_id': int(new_id),
                                  # 'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),#cyc(为了弥补yolov5和cityperson的groundtruth不合)
                                  'category_id': 1,
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    if not in_iter:
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

        # Print results per class
        if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Print speeds
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
        if not training:
            print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # save MR
    if MR:
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        if Datasets == 'cityperson' or Datasets == 'caltechtocity':
            MRs = validate('./val_gt.json', pred_json)#cyc
        if Datasets == 'caltech' or Datasets == 'citytocaltech':
            MRs = validate('./val_caltech.json',pred_json)
        if Datasets == 'foggycity' or Datasets == 'citytofoggy':
            MRs = validate('./val_foggy_city.json', pred_json)
        if Datasets == 'bdd_day' or Datasets == 'citytobdd_day':
            MRs = validate('./val_bdd100.json', pred_json)
        if Datasets == 'bdd_night' or Datasets == 'citytobdd_night':
            MRs = validate('./val_bdd100_night.json', pred_json)
        if Datasets == 'bdd10k':
            MRs = validate('./val_bdd10.json', pred_json)
        # if Datasets == 'citytocaltech':
        #     MRs = validate('./val_caltech_new.json', pred_json)
        # if Datasets == 'caltechtocity':
        #     MRs = validate('./val_city_new.json', pred_json)
        if not in_iter:
            print('Summarize: [Reasonable: %.2f%%], [Bare: %.2f%%], [Partial: %.2f%%], [Heavy: %.2f%%]'
              % (MRs[0] * 100, MRs[1] * 100, MRs[2] * 100, MRs[3] * 100))

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        if not in_iter:
            print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    if MR:
        return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t, MRs
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')#./runs/trainforpaper_crossdomain/exp24/weights/best_mr.pt#./runs/train_forpaper/exp17/weights/best_mr.pt
    # parser.add_argument('--weights', nargs='+', type=str, default='./runs/trainforpaper_crossdomain_transformer/exp34/weights/best_mr.pt', help='model.pt path(s)')#_mr
    # parser.add_argument('--weights', nargs='+', type=str,
    #                     default='./runs/trainforpaper_cycle_baseradar/exp9/weights/best_mr.pt',
    #                     help='model.pt path(s)')
    # parser.add_argument('--weights', nargs='+', type=str,
    #                     default='./runs/train_forpaper/exp28/weights/best_mr.pt',
    #                     help='model.pt path(s)')
    # parser.add_argument('--weights', nargs='+', type=str,
    #                     default='./runs/train_forpaper/exp17/weights/best_mr.pt',
    #                     help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str,
                        default='./runs/trainforpaper_crossdomain_transformer/exp32/weights/best_mr.pt',
                        help='model.pt path(s)')

    # parser.add_argument('--weights', nargs='+', type=str,
    #                     default='./runs/trainforpaper_crossdomain_onlycvt_ctob/exp/weights/best_mr.pt',
    #                     help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/citytocaltech.yaml', help='*.data path')#citytocaltech#cityperson#citytobdd_day#foggycity
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', default=False, help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--pattern', type=str, default='citytocaltech', help='use what dict')#citytocaltech#foggycity
    parser.add_argument('--vision_feature', action='store_true', default=True, help='use what dict')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             Datasets=opt.pattern,
             vision_show_feature=opt.vision_feature,
             img_size= opt.img_size
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
