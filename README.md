# trt_pose_edited
A modified repository from the trt_pose github by NVidia 
Result model compare to 2 pretrained model provided by NVidia
Pretrained model resnet_18_trt 224x224
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.185
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.365
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.164
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.079
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.329
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.245
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.410
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.241
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.083
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.467
 Frame per second   (FPS) = 16

Pretrained model densenet_121_trt 256x256
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.245
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.453
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.236
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.136
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.401
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.314
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.501
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.144
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.546
 Frame per second   (FPS) = 9.5

Modified model dla34_epoch249_upsample_x2_adam_opti_trt cmap_threshold=0.2,link_threshold=0.2 320x320
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.280
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.524
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.260
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.173
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.434
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.347
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.564
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.345
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.181
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.574
 Frame per second   (FPS) = 13.5
