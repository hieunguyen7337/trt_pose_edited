import os
import time
import json
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import trt_pose.coco
import trt_pose.models
import torch
from PIL import Image

WIDTH = 224
HEIGHT = 224
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

# OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
# OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160.pth'
OPTIMIZED_MODEL = 'dla34_epoch249.pth'

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])
    print(OPTIMIZED_MODEL)

    with open('human_pose.json', 'r') as f:
        human_pose = json.load(f)

    topology = trt_pose.coco.coco_category_to_topology(human_pose)

    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])
    
    # model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    # model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    model = trt_pose.models.dla34up_pose(num_parts, 2 * num_links).cuda().eval()
    # model_trt = TRTModule()
    model.load_state_dict(torch.load(OPTIMIZED_MODEL))

    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)

def preprocess(image):
    global device
    device = torch.device('cuda')

    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def process_img(image):
    ori_image = image.copy()
    image = cv2.resize(image, (WIDTH, HEIGHT))
    data = preprocess(image)
    start = time.time()
    start_model = time.time()
    cmap, paf = model(data)
    print("FPS model: ", 1.0/(time.time() - start_model))
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    print("FPS: ", 1.0/(time.time() - start))
    draw_objects(ori_image, counts, objects, peaks)
    return ori_image

def predict_image(path = '1.jpg'):
    image = cv2.imread(path)
    img = process_img(image)
    #cv2.imshow("as", img)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    #cv2.imwrite('predicted_image_resnet_18_2.jpg', img)

def predict_video(path_video):
    print(path_video)
    if os.path.exists(path_video):
        print("exist path video")
        vid = cv2.VideoCapture(path_video)
        frame_width = int(vid.get(3))
        frame_height = int(vid.get(4))
        out = cv2.VideoWriter('predicted_video_dla34.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        while(True):
            ret, frame = vid.read() 
            if not ret:
                # print("no frame")
                break
            
            frame = process_img(frame)
            out.write(frame)
            #frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
	    #cv2.imshow("as", frame)
            #if cv2.waitKey(25) & 0xFF == ord('q'):
            #    break
            
        vid.release()
        cv2.destroyAllWindows() 

#predict_image('2.jpg')
predict_video('test_video.mp4')
