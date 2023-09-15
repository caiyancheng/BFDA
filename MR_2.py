import os
from pycocotools.coco import COCO
from eval_MR_multisetup import COCOeval

def validate(annFile, dt_path):
    mean_MR = []
    for id_setup in range(0, 4):
        cocoGt = COCO(annFile)#标注
        cocoDt = cocoGt.loadRes(dt_path)#预测
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        mean_MR.append(cocoEval.summarize_nofile(id_setup))
    return mean_MR