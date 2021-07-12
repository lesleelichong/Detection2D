# encoding: utf-8
"""
@author: Chong Li
@time: 2021/06/14 18:39
@desc:
"""

import torch
from torch import nn

from networks.backbone.build_backbone import build_backbone
from networks.neck.build_neck import build_neck
from networks.head.build_head import build_head
from networks.nms.build_nms import build_nms
from networks.rpn.build_rpn import build_rpn
from utils.common_utils import resume_weights


__all__ = ["Detector"]


class Detector(nn.Module):
    r"""
    Implements FPN Faster R-CNN or YOLOV3.
    if rpn is None, this is YOLOV3, else, FPN Faster R-CNN.
    """
    def __init__(self, name, backbone, neck, head, nms=None, rpn=None, backbone_resume_from=None):
        super(Detector, self).__init__()
        self.name = name
        self.rpn = None
        self.nms = None
        if rpn is not None:
            self.rpn = build_rpn(rpn)
        if nms is not None:
            self.nms = build_nms(nms)  
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)
        self.init_weights()
        resume_weights(self.backbone, backbone_resume_from)

    def init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs, is_training=False):
        if self.rpn is not None:
            res = self._forward_two_stage(inputs, is_training)
        else:
            res = self._forward_one_stage(inputs, is_training)
        if is_training == False and self.nms is not None:
            with torch.no_grad():
                res = self.nms.process(res)
        return res
    
    def _forward_two_stage(self, inputs, is_training):
        r"""
        Arguments:
            inputs (Dict): use dict to income input parameters.
            if is_training is True, inputs dict must include "image, label, bboxes_num.
        """
        features = self.backbone(inputs['image'])
        features = self.neck(features)
        if is_training:
            self.rpn.incoming_label(inputs)
            self.head.incoming_label(inputs)
        rpn_res = self.rpn(features, is_training=is_training)
        rpn_res['features'] = features
        res = self.head(rpn_res, is_training=is_training)
        return res
    
    def _forward_one_stage(self, inputs, is_training):
        r"""
        Arguments:
            inputs (Dict): use dict to income input parameters.
            for one stage detector, only use inputs["images"].
        """
        features = self.backbone(inputs['image'])
        features = self.neck(features)
        res = self.head(features, is_training=is_training)
        return res
