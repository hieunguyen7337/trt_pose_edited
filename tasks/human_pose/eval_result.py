import pycocotools.coco
import pycocotools.cocoeval
import os
import torch
import torchvision
import trt_pose.coco
import json

cocoGt = pycocotools.coco.COCO('/home/ubuntu/hieunn/trt_pose/trt_pose/annotations/person_keypoints_val2017.json')

catIds = cocoGt.getCatIds('person')
imgIds = cocoGt.getImgIds(catIds=catIds)

cocoDt = cocoGt.loadRes('results_resnet18_epoch249.json')

cocoEval = pycocotools.cocoeval.COCOeval(cocoGt, cocoDt, 'keypoints')
cocoEval.params.imgIds = imgIds
cocoEval.params.catIds = [1]
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
