import json
import trt_pose.coco

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

import torch

# We could then load the saved model using torch2trt as follows.

WIDTH = 224
HEIGHT = 224

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

from torch2trt import TRTModule

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

# We can benchmark the model in FPS with the following code

import time

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))

# Next, let's define a function that will preprocess the image, which is originally in BGR8 / HWC format.

import cv2
import torchvision.transforms as transforms
import PIL.Image

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

# Next, we'll define two callable classes that will be used to parse the objects from the neural network, 
# as well as draw the parsed objects on an image.

from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

# Assuming you're using NVIDIA Jetson, 
# you can use the jetcam package to create an easy to use camera that will produce images in BGR8/HWC format.
# If you're not on Jetson, you may need to adapt the code below.

from jetcam.usb_camera import USBCamera
# from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg

camera = USBCamera(width=WIDTH, height=HEIGHT, capture_fps=30)
# camera = CSICamera(width=WIDTH, height=HEIGHT, capture_fps=30)

camera.running = True

# Next, we'll create a widget which will be used to display the camera feed with visualizations.

import ipywidgets
from IPython.display import display

image_w = ipywidgets.Image(format='jpeg')

display(image_w)

# Finally, we'll define the main execution loop. This will perform the following steps
# 1.Preprocess the camera image
# 2.Execute the neural network
# 3.Parse the objects from the neural network output
# 4.Draw the objects onto the camera image
# 5.Convert the image to JPEG format and stream to the display widget

def execute(change):
    image = change['new']
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    image_w.value = bgr8_to_jpeg(image[:, ::-1, :])

# If we call the cell below it will execute the function once on the current camera frame.

execute({'new': camera.value})

# Call the cell below to attach the execution function to the camera's internal value. 
# This will cause the execute function to be called whenever a new camera frame is received.

camera.observe(execute, names='value')

# Call the cell below to unattach the camera frame callbacks.

camera.unobserve_all()