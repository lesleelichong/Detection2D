# Detection2D: A PyTorch-Based 2D Obejct Detection Framework for FPN Faster R-CNN and YOLOv3

## Major structures

- **Modular code design**

  We decompose the Detection framework into backbone/neck/rpn/head/loss/nms components. one can easily construct a new customized component by using following unified function.
  - build_backbone
  - build_neck
  - build_rpn
  - build_head
  - build_loss
  - build_nms

- **Config design** 

  We use config yaml file to construct the whole training/inference algorithm pipeline.The training/inference formats of different networks are basically the same, which can refer to：
  - configure/train_local/voc/fpn_rcnn.yaml
  - configure/train_local/voc/yolo.yaml
  - configure/infer/voc/fpn_rcnn.yaml
  - configure/infer/voc/yolo.yaml

- **Netwrok Data Input**

  we use dict structure "inputs" to income input parameters.
  - inputs["image"]: Tensor with [batch_size, 3, Height, Width]
  - inputs["label"]: Tensor with [batch_size, max_num, 5(xmin,ymin,xmax,ymax,cls_label)]
  - inputs['bboxes_num']: Batch_size Length List with true gt bboxes_num every image

- **Loss function design**

  Now only support basic loss function like ce_loss, mse, smooth_l1_loss and focal loss.

- **Metric design**

  MAP metric for YOLOv3 and FPN-Faster R-CNN. We add Prec/Recall metric for rpn/rcnn intermediate proposals to analyze the more detailed performance of the detector.


## Getting started

### Installation
- Python 3.8+
- PyTorch 1.7+
- opencv-python

### Training example and Inference example

  It is easy to learn how to use it when we look at train_val/inference python file.
