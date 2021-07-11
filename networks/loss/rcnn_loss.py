# encoding: utf-8
"""
@author: Chong Li
@time: 2021/05/27 18:39
@desc:
"""

import torch
from torch.nn import functional as F
from networks.loss.focal_loss import Binary_Sigmoid_FocalLoss


def smooth_l1_loss(input, target, beta = 1. / 9, size_average = False):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    #return loss.sum()
    return loss


class RcnnLoss(object):
    def __init__(self, rpn_cls_weight=1.0, rpn_box_weight=1.0, 
                       rcnn_cls_weight=1.0, rcnn_box_weight=1.0,
                       rpn_focal_loss=None):
        super(RcnnLoss, self).__init__()
        self.rpn_cls_weight = rpn_cls_weight
        self.rpn_box_weight = rpn_box_weight
        self.rcnn_cls_weight = rcnn_cls_weight
        self.rcnn_box_weight = rcnn_box_weight
        if rpn_focal_loss is not None:
            alpha = rpn_focal_loss.get('alpha', 0.25)
            gamma = rpn_focal_loss.get('gamma', 0.25)
            reduction = rpn_focal_loss.get('reduction', 'mean')
            self.rpn_focal_loss = Binary_Sigmoid_FocalLoss(alpha, gamma, reduction)
        else:
            self.rpn_focal_loss = None
       
    def _compute_rpn_loss(self, inputs):
        props = inputs['props']
        props_targets = inputs['props_targets']
        props_label = inputs['props_label']
        props_score = inputs['props_score']
        if self.rpn_focal_loss is None:
            cls_loss = F.binary_cross_entropy_with_logits(props_score, props_label)
        else:
            cls_loss = self.rpn_focal_loss(props_score, props_label)
        box_loss = F.smooth_l1_loss(props, props_targets, reduction='none', beta=1. / 9)
        box_loss = box_loss.sum() / len(props_label)
        return cls_loss, box_loss
    
    def _compute_rcnn_loss(self, inputs):
        props = inputs['props']
        props_targets = inputs['props_targets']
        props_label = inputs['props_label']
        props_score = inputs['props_score']
        props_label = props_label.long()
        cls_loss = F.cross_entropy(props_score, props_label)
        
        sampled_pos_inds = torch.where(props_label > 0)[0]
        labels_pos = props_label[sampled_pos_inds]
        props = props.reshape(len(props_targets), -1, 4)
        props_pos = props[sampled_pos_inds, labels_pos]
        props_pos_targets = props_targets[sampled_pos_inds]
        box_loss = F.smooth_l1_loss(props_pos, props_pos_targets, reduction='none', beta=1. / 9)
        box_loss = box_loss.sum() / props_label.numel()
        return cls_loss, box_loss
           
    def compute_loss(self, inputs):
        inputs = inputs['pred']
        rpn_cls_loss, rpn_box_loss = self._compute_rpn_loss(inputs['rpn'])
        rcnn_cls_loss, rcnn_box_loss = self._compute_rcnn_loss(inputs)
        total_loss = (self.rpn_cls_weight * rpn_cls_loss + 
                      self.rpn_box_weight * rpn_box_loss +
                      self.rcnn_cls_weight * rcnn_cls_loss +
                      self.rcnn_box_weight * rcnn_box_loss)
        return {
            "rpn_cls_loss" : self.rpn_cls_weight * rpn_cls_loss , 
            "rpn_box_loss" : self.rpn_box_weight * rpn_box_loss,
            "rcnn_cls_loss" : self.rcnn_cls_weight * rcnn_cls_loss , 
            "rpn_box_loss" : self.rcnn_box_weight * rcnn_box_loss,
            "total" : total_loss}