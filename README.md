# trt_pose_edited
A modified repository from the trt_pose github by NVidia 

Modified model result compare to the two pretrained model provided by NVidia
| Model | Pretrained model resnet_18_trt 224x224 | Pretrained model densenet_121_trt 256x256 | Modified model dla34_epoch249_upsample_x2_adam_opti_trt cmap_threshold=0.2,link_threshold=0.2 320x320 |
|-------|-------------|---------------|---------|
| Average Precision (AP) @[IoU=0.05:0.50:0.95] | 0.185 | 0.245 | 0.280 |
| Average Precision (AP) @[IoU=0.50]           | 0.365 | 0.453 | 0.524 |
| Average Precision (AP) @[IoU=0.75]           | 0.164 | 0.236 | 0.260 |
| Average Precision (AP) @[area=medium]        | 0.079 | 0.136 | 0.173 |
| Average Precision (AP) @[area= large]        | 0.329 | 0.401 | 0.434 |
| Average Recall    (AR) @[IoU=0.05:0.50:0.95] | 0.245 | 0.314 | 0.347 |
| Average Recall    (AR) @[IoU=0.50]           | 0.410 | 0.501 | 0.564 |
| Average Recall    (AR) @[IoU=0.75]           | 0.241 | 0.320 | 0.345 |
| Average Recall    (AR) @[area=medium]        | 0.083 | 0.144 | 0.181 |
| Average Recall    (AR) @[area= large]        | 0.467 | 0.546 | 0.574 |
| Frame per second  (FPS)                      | 16 | 9.5 | 13.5 |

All evaluation perform using the COCO human pose validation dataset and the FPS are measured by using NVidia Jetson Nano.
