import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread
from PIL import Image
import matplotlib.pyplot as plt
from show import show_feature
from deal import *
from Cheat import *

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr, non_max_suppression, xyxy2xywh
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from domain_ad import *
from zb_discriminator import *
from torchvision import *
from torch.nn import functional as F
from feature_distill_simple import F_D
from cycle_generator import get_F_G

# from vit_pytorch import ViT
# from vit_pytorch.cvt import CvT

from sam import SAM
SHOW_F = False
look_iter = 100

logger = logging.getLogger(__name__)
#图像的格式变换
unloader = transforms.ToPILImage()
loader = transforms.Compose([transforms.ToTensor()])

Train_learning_rate_LSD = 1e-4
Train_learning_rate_BDM = 1e-4
Train_learning_rate_FGM = 1e-4

Train_GAN_loss = 'BCE'
dis_loss = bce_loss if Train_GAN_loss == 'BCE' else ls_loss

Train_loss_prop_det = 1.
# Train_loss_prop_gen = 0.05
Train_loss_prop_dis = 0.001

Train_dataset_augment =  False

LEVEL = 3#在哪一阶段的输出使用图像级域对齐(backbone?neck?detect?) backbone——1  neck——2  detect——3

TRAIN_LAYER = 1 #一般只训练前两层相对比较合适尤其是背景部分

SHOW = True

model_trans = 'cvt_hr' #cvt#vit#cvt_official
IF_SC_dis = False



def train(hyp, opt, device, tb_writer=None, MR=True):
    pattern = 'debug' if opt.project.endswith('debug') else 'train'
    print(pattern)
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank
    wdir = save_dir / 'weights'#runs/train_crossdomain/exp/weights
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'#runs/train_crossdomain/exp/weights/last.pt
    best = wdir / 'best.pt'#runs/train_crossdomain/exp/weights/best.pt
    best_mr = wdir / 'best_mr.pt'
    results_file = save_dir / 'results.txt'#runs/train_crossdomain/exp/results.txt

    # Save run settings保存各种参数数据
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots#T
    cuda = device.type != 'cpu'#T
    init_seeds(2 + rank)#1
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = opt.data.endswith('coco.yaml')

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict#读入cityperson.yaml的东西
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names['pedestrain]
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint加载检查点
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create#此处进入yolo的框架
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']
    ##########################################cyc
    target_train_path = data_dict['target_train']
    # target_test_path = data_dict['target_val']
    ##########################################
    # Freeze
    #冻结模型层, 设置冻结层名字即可
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    """
        nbs为模拟的batch_size; 
        就比如默认的话上面设置的opt.batch_size为16,这个nbs就为64，
        也就是模型梯度累积了64/16=4(accumulate)次之后
        再更新一次模型，变相的扩大了batch_size
        """
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing(3)
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    # 将模型分成三组(weight、bn, bias, 其他所有参数)优化
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    # 选用优化器，并设置pg0组的优化方式
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    # 设置weight、bn权重的优化方式
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    #设置bias的优化方式
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:#学习率的变化
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']#余弦退火衰减
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA# 为模型创建EMA指数滑动平均,如果GPU进程数大于1,则不创建#############################################################################
    ema = ModelEMA(model) if rank in [-1, 0] else None
    LSD_model = get_model(model=model_trans)
    # BDM_model = F_D(input_channel=18)
    # FGM_model = get_F_G()
    LSD_model.train()
    # BDM_model.train()
    # FGM_model.train()


    # Resume
    start_epoch, best_fitness = 0, 0.0
    best_MR = 100.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        # start_epoch = ckpt['epoch'] + 1
        start_epoch, best_fitness = 0, 0.0
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)#gs:64
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])#nl:4
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        LSD_model = torch.nn.DataParallel(LSD_model)
        BDM_model = torch.nn.DataParallel(BDM_model)
        FGM_model = torch.nn.DataParallel(FGM_model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device,non_blocking=True)
        LSD_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(LSD_model).to(device,non_blocking=True)
        BDM_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(BDM_model).to(device,non_blocking=True)
        FGM_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(FGM_model).to(device,non_blocking=True)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad,
                                            prefix=colorstr('train: '))
    dis_src_train_loader, _ = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                                hyp=hyp, augment=Train_dataset_augment, cache=opt.cache_images, rect=opt.rect,
                                                rank=rank,
                                                world_size=opt.world_size, workers=opt.workers,
                                                image_weights=opt.image_weights, quad=opt.quad,
                                                prefix=colorstr('source_train: '))
    dis_trg_train_loader, _ = create_dataloader(target_train_path, imgsz_test, batch_size, gs, opt,
                                                hyp=hyp, augment=Train_dataset_augment, cache=opt.cache_images, rect=opt.rect,
                                                rank=rank,
                                                world_size=opt.world_size, workers=opt.workers,
                                                image_weights=opt.image_weights, quad=opt.quad,
                                                prefix=colorstr('target_train: '))

###########################################################################################################################
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class获取标签最大的类别值，用以检查是否有错误
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        if opt.iter_test:
            batch_size_test = 8
        else:
            batch_size_test = batch_size*2
        testloader = create_dataloader(test_path, imgsz_test, batch_size_test, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))
        LSD_model = DDP(LSD_model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))
        # BDM_model = DDP(BDM_model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))
        # FGM_model = DDP(FGM_model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))


    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = 0
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    optimizer_LSD = optim.Adam(LSD_model.parameters(), lr=Train_learning_rate_LSD, betas=(0.9, 0.99))
    # optimizer_BDM = optim.Adam(BDM_model.parameters(), lr=Train_learning_rate_BDM, betas=(0.9, 0.99))
    # optimizer_FGM = optim.Adam(FGM_model.parameters(), lr=Train_learning_rate_FGM, betas=(0.9, 0.99))

    source_label = 0
    target_label = 1

    resize_c = nn.Sequential(nn.Conv2d(18, 3, kernel_size=3, stride=1, padding=1),
                             nn.LeakyReLU(negative_slope=0.2, inplace=True)).to(device).half()

#######################################################################
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        LSD_model.train()
        # BDM_model.train()
        # FGM_model.train()

        # Update image weights (optional)
        if opt.image_weights:  # False
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
            dis_src_train_loader.sampler.set_epoch(epoch)
            dis_trg_train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(zip(dataloader, dis_src_train_loader, dis_trg_train_loader))
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        nb = min(len(dataloader),len(dis_src_train_loader),len(dis_trg_train_loader))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        optimizer_LSD.zero_grad()
        # optimizer_BDM.zero_grad()
        # optimizer_FGM.zero_grad()
###########################################################下面就是按照iter计数

        # for i, (imgs, targets, paths, _),_, batch in pbar :  # batch -------------------------------------------------------------
        for i, (src,dis_src,dis_trg) in pbar:#源域和目标域
            LSD_model = LSD_model.to(device)#LSD_model.train()
            # BDM_model = BDM_model.to(device)
            # FGM_model = FGM_model.to(device)
            imgs, real_labels, paths, _ = src#imgs:[b,3,2048,2048],labels:[16,6] #这6个:[第几张图，0，坐标]
            imgs_src, real_labels_src, paths_src, _ = dis_src
            imgs_trg, _, paths_trg, _ = dis_trg

            src_show_i = find_pict(pict_name=os.path.join(train_path, 'aachen/aachen_000031_000019_leftImg8bit.png'),
                                       paths=paths)
            trg_show_i = find_pict(pict_name=os.path.join(target_train_path, 'set00/set00_V000_242.png'),  # _small
                                       paths=paths_trg)

            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            imgs_src = imgs_src.to(device, non_blocking=True).float() / 255.0
            if SHOW and epoch == 0 and src_show_i >= 0:
                src_img_show = unloader(imgs_src[src_show_i].squeeze(0))
                cv2.imwrite(str(save_dir)+'/src_img_aachen_000031_000019.png', np.array(src_img_show)[..., ::-1])

            # Warmup
            if ni <= nw: #1000个it做warmup
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())#2.0
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Forward(Only train yolo)
            ################################# 1、yoloV5正常训练源域图片#############################
            LSD_model = deal_grad(LSD_model, key=False)
            # BDM_model = deal_grad(BDM_model, key=False)
            # FGM_model = deal_grad(FGM_model, key=False)

            with amp.autocast(enabled=cuda):
                _, pred, feature_src, feature_afterconv_src, feature_afterbackbone_src = model(imgs)
                loss, loss_items = compute_loss(pred, real_labels.to(device, non_blocking=True))#4:0.10820  # loss scaled by batch_size #back_prob = get_front_or_back(pred)
                if opt.quad:
                    loss *= 4.

            scaler.scale(loss * Train_loss_prop_det).backward()
            # if ni%look_iter == 0:
            #     print('base yolo loss:',float(loss))

            # #看下源域图片的特征图
            if src_show_i>=0 and SHOW_F:
                show_feature(feature = feature_afterconv_src,img_id=src_show_i,img_size=opt.img_size[0],save_dir=save_dir,epoch=epoch,remark='real_src')

            # src_back_cycle = paste_label(imgs=imgs_src.clone(), real_labels=real_labels_src, all_black=False)
            # src_back_cycle = F.interpolate(src_back_cycle,size=(256,256),mode='bilinear')
            # src_back_cycle = src_back_cycle.to(device)

            # if SHOW and epoch == 0 and src_show_i>=0:
            #     src_img_show = unloader(src_back_cycle[src_show_i].squeeze(0))
            #     cv2.imwrite(str(save_dir) + '/src_img_aachen_000031_000019_src_cycle_label.png', np.array(src_img_show)[..., ::-1])#黑白图片就不能再转这个了

            ################################## 2、源域图片第一次前向传播#############################
            with amp.autocast(enabled=cuda):
                _, pred, _, feature_afterconv_src, feature_afterbackbone_src = model(imgs_src)
                #loss_variance = compute_loss_variance(feature=feature_afterconv_src,control_height=False,Layer=2)*TRAIN_LEARNING_RATE_D_Variance
            feature_src = feature_afterconv_src

            #loss_SRC = loss_variance
            loss_SRC = 0

            ################################## 3、欺骗判别器（epoch>=1时）#############################
            Cheat = wetherCheat(epoch=epoch,MY_14=True)
            imgs_trg = imgs_trg.to(device, non_blocking=True).float() / 255.
            # imgs_trg_new_back = imgs_trg.clone()
            if SHOW and epoch == 0 and trg_show_i >= 0:
                trg_img_show = unloader(imgs_trg[trg_show_i].squeeze(0))
                cv2.imwrite(str(save_dir) + '/trg_img_set00_V000_242.png', np.array(trg_img_show)[..., ::-1])
            with amp.autocast(enabled=cuda):
                out_trg, pred_trg, _, feature_afterconv_trg, feature_afterbackbone_trg = model(imgs_trg)
                # out_trg_NMS = non_max_suppression(out_trg, conf_thres=0.01, iou_thres=0.6, labels=[], multi_label=True)
                # trg_back_cycle = paste_trg_preds(imgs=imgs_trg_new_back, out_nms=out_trg_NMS, all_black=False)
                # trg_back_cycle = F.interpolate(trg_back_cycle, size=(256, 256), mode='bilinear')
                # trg_back_cycle = trg_back_cycle.to(device, non_blocking=True)

            # if SHOW and epoch == 0 and trg_show_i >= 0:
            #     trg_img_show = unloader(trg_back_cycle[trg_show_i].squeeze(0))
            #     cv2.imwrite(str(save_dir) + '/trg_img_set00_V000_242_background_trg_cycle_label.png', np.array(trg_img_show)[..., ::-1])
            feature_trg = feature_afterconv_trg

            if trg_show_i >= 0 and SHOW_F:
                show_feature(feature=feature_afterconv_trg, img_id=trg_show_i, img_size=opt.img_size[1],
                             save_dir=save_dir, epoch=epoch, remark='real_trg')
            # with amp.autocast(enabled=cuda):
            #     feature_src_B = BDM_model(feature_src[0])
            #     feature_trg_B = BDM_model(feature_trg[0])
            feature_src[0] = resize_c(F.interpolate(feature_src[0], size=(224, 224), mode='bilinear', align_corners=True))
            feature_trg[0] = resize_c(F.interpolate(feature_trg[0], size=(224, 224), mode='bilinear', align_corners=True))
            if Cheat: #让目标域的图片被认为来自源域
                with amp.autocast(enabled=cuda):
                    d_out_first = LSD_model(feature_trg[0])
                    loss_adv_trg_first = dis_loss(d_out_first, source_label)
                    loss_adv = Train_loss_prop_dis * loss_adv_trg_first #*0.03
                    scaler.scale(loss_adv).backward()
                    # if ni%look_iter == 0:
                    #     print('cheat loss(T-->S):', float(loss_adv))

                    d_out_first_src = LSD_model(feature_src[0])
                    loss_adv_src_first = dis_loss(d_out_first_src, target_label)
                    loss_adv_src = Train_loss_prop_dis * loss_adv_src_first #*0.03
                    loss_SRC += loss_adv_src
                    # if ni%look_iter == 0:
                    #     print('cheat loss(S-->T):', float(loss_adv_src))
            if loss_SRC:
                scaler.scale(loss_SRC).backward()

            ################################## 4、训练cycle的consistency#############################
            # BDM_model = deal_grad(BDM_model, key=True)
            # FGM_model = deal_grad(FGM_model, key=True)
            # with amp.autocast(enabled=cuda):
            #     feature_src_B = BDM_model(feature_src[0].detach())
            #     feature_trg_B = BDM_model(feature_trg[0].detach())
            #     feature_src_G = FGM_model(feature_src_B)
            #     feature_trg_G = FGM_model(feature_trg_B)
            # with amp.autocast(enabled=cuda):
            #     loss_cycle_src = nn.L1Loss()(feature_src_G, src_back_cycle)
            #     loss_cycle_trg = nn.L1Loss()(feature_trg_G, trg_back_cycle)
            #     loss_cycle = loss_cycle_src + loss_cycle_trg
            #     scaler.scale(loss_cycle * Train_loss_prop_gen).backward()
            # if SHOW and src_show_i >= 0:
            #     src_img_show = unloader(feature_src_G[src_show_i].squeeze(0))
            #     cv2.imwrite(str(save_dir)+'/src_img_aachen_000031_000019_cycle_ge_E'+str(epoch)+'.png', np.array(src_img_show)[..., ::-1])
            # if SHOW and trg_show_i >= 0:
            #     trg_img_show = unloader(feature_trg_G[trg_show_i].squeeze(0))
            #     cv2.imwrite(str(save_dir)+'/trg_img_set00_V000_242_cycle_ge_E'+str(epoch)+'.png', np.array(trg_img_show)[..., ::-1])

            LSD_model = deal_grad(LSD_model, key=True)
            # BDM_model = deal_grad(BDM_model, key=False)
            # FGM_model = deal_grad(FGM_model, key=False)
            with amp.autocast(enabled=cuda):
                feature_src[0] = feature_src[0].detach()
                d_out_first = LSD_model(feature_src[0])
                loss_d_first = dis_loss(d_out_first, source_label) #0.13989
                loss_d_first = loss_d_first * Train_loss_prop_dis
                scaler.scale(loss_d_first).backward()
                # if ni%look_iter  == 0:
                #     print('source: loss_d_first\t',float(loss_d_first))

                feature_trg[0] = feature_trg[0].detach()
                d_out_first = LSD_model(feature_trg[0])
                loss_d_first = dis_loss(d_out_first, target_label)#0.94167
                loss_d_first = loss_d_first * Train_loss_prop_dis
                scaler.scale(loss_d_first).backward()
                # if ni%look_iter == 0:
                #     print('target: loss_d_first\t',float(loss_d_first))
                    # print('target first layer:\t ', torch.sigmoid_(d_out_first))


             # Optimize
            if ni % accumulate == 0:
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.step(optimizer_LSD)
                # scaler.step(optimizer_BDM)
                # scaler.step(optimizer_FGM)
                scaler.update()
                optimizer.zero_grad()
                optimizer_LSD.zero_grad()
                # optimizer_BDM.zero_grad()
                # optimizer_FGM.zero_grad()
                if ema:
                    ema.update(model.to(device, non_blocking=True))

            if opt.iter_test and epoch > 0:
                if ni % 200 == 0:
                    results, maps, times, MRs = test.test(data_dict,
                                                          batch_size=8,
                                                          imgsz=imgsz_test,
                                                          model=ema.ema,
                                                          single_cls=opt.single_cls,
                                                          dataloader= testloader,
                                                          save_dir=save_dir,
                                                          verbose=False,
                                                          plots=False,
                                                          wandb_logger= wandb_logger,
                                                          compute_loss=compute_loss,
                                                          Datasets='foggycity',
                                                          is_coco=is_coco, MR=True, vision_show_feature= False, iter=i,
                                                          epoch=epoch, in_iter=True)
                    print('epoch:  '+str(epoch)+'\t'+'iter:  '+str(i)+ '\t'+ 'MR_reasonable:  '+ str(MRs[0]*100))
                    if MRs[0] < best_MR:
                        best_MR = MRs[0]
                        ckpt = {'epoch': epoch,
                                'best_fitness': best_fitness,
                                'training_results': results_file.read_text(),
                                'model': deepcopy(model.module if is_parallel(model) else model).half(),
                                'ema': deepcopy(ema.ema).half(),
                                'updates': ema.updates,
                                'optimizer': optimizer.state_dict(),
                                'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                        torch.save(ckpt, best_mr)
                        del ckpt
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, real_labels.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 3:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, real_labels, paths, f), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(model, imgs)  # add model to tensorboard
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------
        # print("epoch:",epoch)
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()
        # if IF_SC_dis:
        #     scheduler_d_first.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                if MR:
                    results, maps, times, MRs = test.test(data_dict,
                                                          batch_size=batch_size_test,
                                                          imgsz=imgsz_test,
                                                          model=ema.ema,
                                                          single_cls=opt.single_cls,
                                                          dataloader=testloader,
                                                          save_dir=save_dir,
                                                          verbose=nc < 50 and final_epoch,
                                                          plots=plots and final_epoch,
                                                          wandb_logger=wandb_logger,
                                                          compute_loss=compute_loss,
                                                          Datasets='foggycity',
                                                          is_coco=is_coco, MR=True)
                else:
                    results, maps, times = test.test(data_dict,
                                                     batch_size=batch_size_test,
                                                     imgsz=imgsz_test,
                                                     model=ema.ema,
                                                     single_cls=opt.single_cls,
                                                     dataloader=testloader,
                                                     save_dir=save_dir,
                                                     verbose=nc < 50 and final_epoch,
                                                     plots=plots and final_epoch,
                                                     wandb_logger=wandb_logger,
                                                     compute_loss=compute_loss,
                                                     is_coco=is_coco, MR=False)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            new_mr = MRs[0]
            if new_mr < best_MR:
                best_MR = new_mr

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if best_MR == new_mr:
                    torch.save(ckpt, best_mr)
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
            for m in (last, best) if best.exists() else (last):  # speed, mAP tests
                if MR:
                    results, _, _, MRs = test.test(opt.data,
                                                   batch_size=batch_size_test,
                                                   imgsz=imgsz_test,
                                                   conf_thres=0.001,
                                                   iou_thres=0.7,
                                                   model=attempt_load(m, device).half(),
                                                   single_cls=opt.single_cls,
                                                   dataloader=testloader,
                                                   save_dir=save_dir,
                                                   save_json=True,
                                                   plots=False,
                                                   Datasets='foggycity',
                                                   is_coco=is_coco, MR=True)
                else:
                    results, _, _ = test.test(opt.data,
                                              batch_size=batch_size_test,
                                              imgsz=imgsz_test,
                                              conf_thres=0.001,
                                              iou_thres=0.7,
                                              model=attempt_load(m, device).half(),
                                              single_cls=opt.single_cls,
                                              dataloader=testloader,
                                              save_dir=save_dir,
                                              save_json=True,
                                              plots=False,
                                              is_coco=is_coco, MR=False)

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', type=str, default='./runs/trainforpaper_crossdomain/exp48/weights/best_mr.pt', help='initial weights path')
    parser.add_argument('--weights', type=str, default='./runs/train_tip_cityscapes_2048_mosaic/exp4/weights/best_mr.pt', help='initial weights path')#哪种yolo结构？
    # parser.add_argument('--weights', type=str, default='/remote-home/source/42/cyc19307140030/dellyolo/runs/dell_train/exp7/weights/best.pt',
    #                     help='initial weights path')  # 哪种yolo结构？
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')#参考.yaml文件
    parser.add_argument('--data', type=str, default='data/citytofoggy.yaml', help='data.yaml path')#数据集目录#city_to_foggy#citytocaltech
    # parser.add_argument('--data', type=str, default='data/dell.yaml', help='data.yaml path')  # 数据集目录
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')#超参们
    parser.add_argument('--epochs', type=int, default=20)#几个epoch?
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs')#batchsize
    parser.add_argument('--img-size', nargs='+', type=int, default=[2048, 2048], help='[train, test] image sizes')#切割
    parser.add_argument('--rect',action='store_true',default=True, help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', default=False, help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', default=False,help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/UDA_city2foggy_BFDA_LS', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')#保存到的文件名
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--iter_test', action='store_true', default=True,help='test each iter instead of each epoch')
    opt = parser.parse_args()


    # Set DDP variables设置分布式学习
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1#1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1#-1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()
        check_requirements()

    # Resume
    wandb_run = check_wandb_resume(opt)#查找有没有存
    wandb_run = None
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # DDP mode
    # 分布式
    opt.total_batch_size = opt.batch_size#4
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    # 数据增强选项
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')