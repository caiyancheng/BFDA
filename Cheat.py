from domain_ad import *

def cheatbackbone_back(feature,d_1,d_2,d_3,d_4,MY_4,MY_1,TRAIN_GANLOSS,ADD):
    d_out_first = d_1(feature[0])  # d_out_first:[b,2,8,8]
    if not MY_4:  # 使用张铂师兄的方法
        if MY_1:
            source_label = 0.5
        if TRAIN_GANLOSS == 'BCE':
            loss_adv_trg_first = bce_loss(d_out_first, source_label)
        elif TRAIN_GANLOSS == 'LS':
            loss_adv_trg_first = ls_loss(d_out_first, source_label)
        if ADD == 1:
            d_out_first_add = d_first_add(feature_trg[0])  # d_out_first:[24,1,2,2]
            if TRAIN_GANLOSS == 'BCE':
                loss_adv_trg_first_add = bce_loss(d_out_first_add, source_label)
            elif TRAIN_GANLOSS == 'LS':
                loss_adv_trg_first_add = ls_loss(d_out_first_add, source_label)
        if TRAIN_LAYER > 1:
            if ddcpp_2 == 0:
                d_out_second = d_second(feature_trg[1])  # ([24,1,1,1])
            if ddcpp_2 == 1:
                d_out_second = dis_model_2(feature_trg[1])  # ([24,1,1,1])
            if TRAIN_GANLOSS == 'BCE':
                loss_adv_trg_second = bce_loss(d_out_second, source_label)
            elif TRAIN_GANLOSS == 'LS':
                loss_adv_trg_second = ls_loss(d_out_second, source_label)
        if TRAIN_LAYER > 2:
            d_out_third = d_third(feature_trg[2])  # ([24,1,1,1])
            if TRAIN_GANLOSS == 'BCE':
                loss_adv_trg_third = bce_loss(d_out_third, source_label)
            elif TRAIN_GANLOSS == 'LS':
                loss_adv_trg_third = ls_loss(d_out_third, source_label)
        if TRAIN_LAYER > 3:
            d_out_forth = d_forth(feature_trg[3])  # ([24,1,1,1])
            if TRAIN_GANLOSS == 'BCE':
                loss_adv_trg_forth = bce_loss(d_out_forth, source_label)
            elif TRAIN_GANLOSS == 'LS':
                loss_adv_trg_forth = ls_loss(d_out_forth, source_label)
        loss_adv = TRAIN_LAMBADA_ADV_FIRST * loss_adv_trg_first
        if ADD == 1:
            loss_adv = loss_adv + TRAIN_LAMBADA_ADV_FIRST * loss_adv_trg_first_add
        if TRAIN_LAYER > 1:
            loss_adv = loss_adv + TRAIN_LAMBADA_ADV_SECOND * loss_adv_trg_second
        if TRAIN_LAYER > 2:
            loss_adv = loss_adv + TRAIN_LAMBADA_ADV_THIRD * loss_adv_trg_third
        if TRAIN_LAYER > 3:
            loss_adv = loss_adv + TRAIN_LAMBADA_ADV_FORTH * loss_adv_trg_forth
        if MY_6 and not MY_13:
            loss_adv += loss_variance_trg
        loss_adv = loss_adv
        scaler.scale(loss_adv).backward()

    if MY_4:  # 用反梯度
        if MY_1:
            target_label = 0.5
        if TRAIN_GANLOSS == 'BCE':
            loss_adv_trg_first = bce_loss(d_out_first, target_label)
        elif TRAIN_GANLOSS == 'LS':
            loss_adv_trg_first = ls_loss(d_out_first, target_label)
        if ADD == 1:
            d_out_first_add = d_first_add(feature_trg[0])  # d_out_first:[24,1,2,2]
            if TRAIN_GANLOSS == 'BCE':
                loss_adv_trg_first_add = bce_loss(d_out_first_add, target_label)
            elif TRAIN_GANLOSS == 'LS':
                loss_adv_trg_first_add = ls_loss(d_out_first_add, target_label)
        if TRAIN_LAYER > 1:
            if ddcpp_2 == 0:
                d_out_second = d_second(feature_trg[1])  # ([24,1,1,1])
            if ddcpp_2 == 1:
                d_out_second = dis_model_2(feature_trg[1])  # ([24,1,1,1])
            if TRAIN_GANLOSS == 'BCE':
                loss_adv_trg_second = bce_loss(d_out_second, target_label)
            elif TRAIN_GANLOSS == 'LS':
                loss_adv_trg_second = ls_loss(d_out_second, target_label)
        if TRAIN_LAYER > 2:
            d_out_third = d_third(feature_trg[2])  # ([24,1,1,1])
            if TRAIN_GANLOSS == 'BCE':
                loss_adv_trg_third = bce_loss(d_out_third, target_label)
            elif TRAIN_GANLOSS == 'LS':
                loss_adv_trg_third = ls_loss(d_out_third, target_label)
        if TRAIN_LAYER > 3:
            d_out_forth = d_forth(feature_trg[3])  # ([24,1,1,1])
            if TRAIN_GANLOSS == 'BCE':
                loss_adv_trg_forth = bce_loss(d_out_forth, target_label)
        elif TRAIN_GANLOSS == 'LS':
            loss_adv_trg_forth = ls_loss(d_out_forth, target_label)
        loss_adv = TRAIN_LAMBADA_ADV_FIRST * loss_adv_trg_first
        if ADD == 1:
            loss_adv = loss_adv + TRAIN_LAMBADA_ADV_FIRST * loss_adv_trg_first_add
        if TRAIN_LAYER > 1:
            loss_adv = loss_adv + TRAIN_LAMBADA_ADV_SECOND * loss_adv_trg_second
        if TRAIN_LAYER > 2:
            loss_adv = loss_adv + TRAIN_LAMBADA_ADV_THIRD * loss_adv_trg_third
        if TRAIN_LAYER > 3:
            loss_adv = loss_adv + TRAIN_LAMBADA_ADV_FORTH * loss_adv_trg_forth
        loss_adv = - loss_adv
        if MY_6 and not MY_13:
            loss_adv += loss_variance_trg
        loss_adv = loss_adv
        scaler.scale(loss_adv).backward()