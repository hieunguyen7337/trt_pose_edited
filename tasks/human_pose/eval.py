import pycocotools.coco
import pycocotools.cocoeval
import os
import torch
import PIL.Image
import torchvision
import torchvision.transforms
import trt_pose.plugins
import trt_pose.models
import trt_pose.coco
# import torch2trt
import tqdm
import json
# from torch2trt import TRTModule
from trt_pose.parse_objects import ParseObjects
import time

# @tensorrt_module
# class My

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
# model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
# model = trt_pose.models.dla34up_pose(num_parts, 2 * num_links).cuda().eval()

MODEL_WEIGHT = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
# MODEL_WEIGHT = 'densenet121_baseline_att_256x256_B_epoch_160.pth'
# MODEL_WEIGHT = '/home/ubuntu/hieunn/trt_pose/tasks/human_pose/experiments/dla34up_pose_256x256_A_epoch_102.json.checkpoints/epoch_147.pth'

# model = TRTModule()
model.load_state_dict(torch.load(MODEL_WEIGHT))

# import pdb
# pdb.pm()

# model = model.cuda().eval()

# cmap, paf = model(torch.zeros((1, 3, 256, 256)).cuda())

# print(cmap.shape)

# print(paf.shape)

IMAGE_SHAPE = (256, 256)
images_dir = '/home/ubuntu/dataset/coco/val2017'
annotation_file = '/home/ubuntu/hieunn/trt_pose/trt_pose/annotations/person_keypoints_val2017_modified.json'

cocoGtTmp = pycocotools.coco.COCO('/home/ubuntu/hieunn/trt_pose/trt_pose/annotations/person_keypoints_val2017_modified.json')

topology = trt_pose.coco.coco_category_to_topology(cocoGtTmp.cats[1])

cocoGt = pycocotools.coco.COCO('/home/ubuntu/hieunn/trt_pose/trt_pose/annotations/person_keypoints_val2017.json')

catIds = cocoGt.getCatIds('person')
imgIds = cocoGt.getImgIds(catIds=catIds)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

parse_objects = ParseObjects(topology, cmap_threshold=0.05, link_threshold=0.1, cmap_window=11, line_integral_samples=7, max_num_parts=100, max_num_objects=100)

results = []
start_time = time.time()

for n, imgId in enumerate(imgIds):
    
    # read image
    img = cocoGt.imgs[imgId]
    img_path = os.path.join(images_dir, img['file_name'])

    image = PIL.Image.open(img_path).convert('RGB').resize(IMAGE_SHAPE)
    data = transform(image).cuda()[None, ...]

    cmap, paf = model(data)
    cmap, paf = cmap.cpu(), paf.cpu()

#     object_counts, objects, peaks, int_peaks = postprocess(cmap, paf, cmap_threshold=0.05, link_threshold=0.01, window=5)
#     object_counts, objects, peaks = int(object_counts[0]), objects[0], peaks[0]
    
    object_counts, objects, peaks = parse_objects(cmap, paf)
    object_counts, objects, peaks = int(object_counts[0]), objects[0], peaks[0]

    for i in range(object_counts):
        object = objects[i]
        score = 0.0
        kps = [0]*(17*3)
        x_mean = 0
        y_mean = 0
        cnt = 0
        for j in range(17):
            k = object[j]
            if k >= 0:
                peak = peaks[j][k]
                x = round(float(img['width'] * peak[1]))
                y = round(float(img['height'] * peak[0]))
                score += 1.0
                kps[j * 3 + 0] = x
                kps[j * 3 + 1] = y
                kps[j * 3 + 2] = 2
                x_mean += x
                y_mean += y
                cnt += 1

        ann = {
            'image_id': imgId,
            'category_id': 1,
            'keypoints': kps,
            'score': score / 17.0
        }
        results.append(ann)
    if n % 100 == 0:
        print('%d / %d' % (n, len(imgIds)))
        print('time taken per image:',(time.time() - start_time)/100)
        start_time = time.time()
#        break
        
with open('results_resnet18_epoch249.json', 'w') as f:
    json.dump(results, f)

# cocoDt = cocoGt.loadRes('results_resnet18.json')

# cocoEval = pycocotools.cocoeval.COCOeval(cocoGt, cocoDt, 'keypoints')
# cocoEval.params.imgIds = imgIds
# cocoEval.params.catIds = [1]
# cocoEval.evaluate()
# cocoEval.accumulate()
# cocoEval.summarize()
