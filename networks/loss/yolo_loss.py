# encoding: utf-8
"""
@author: Chong Li
@time: 2021/05/27 18:39
@desc:
"""

import numpy as np
import torch
from torch.nn import functional as F
from utils.bbox_utils import BBoxBatchIOU


def GTBoxAnchorIndex(gt_boxes, anchors):
    gt_boxes = gt_boxes.unsqueeze(-2)
    gt_boxes[...,2] = gt_boxes[...,2] - gt_boxes[...,0]
    gt_boxes[...,3] = gt_boxes[...,3] - gt_boxes[...,1]
    gt_boxes[...,0] = 0
    gt_boxes[...,1] = 0
    anchor_boxes = torch.cat([anchors*0, anchors], dim=-1)
    ious = BBoxBatchIOU(gt_boxes, anchor_boxes)
    anchor_index = torch.max(ious, dim=-1)[1]
    return anchor_index.detach().cpu().numpy()
    

class YoloLoss(object):
    def __init__(self, weight_xy=1.0, weight_wh=1.0, weight_obj=1.0, 
                 weight_cls=1.0, 
                 lambda_pos = 1.0,
                 lambda_neg = 0.005,
                 ignore_iou_thresh=0.7):
        super(YoloLoss, self).__init__()
        self.ignore_iou_thresh = ignore_iou_thresh
        self.weight_xy = weight_xy
        self.weight_wh = weight_wh
        self.weight_obj = weight_obj
        self.weight_cls = weight_cls
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg

    def _target_generator(self, y_pred, y_label, gt_bboxes_num):
        center_targets = torch.zeros_like(y_pred['raw_box_centers'])
        scale_targets = torch.zeros_like(y_pred['raw_box_scales'])
        scale_weights = torch.zeros_like(y_pred['raw_box_scales'])
        objness_targets = torch.zeros_like(y_pred['objness_pred'])
        class_targets = torch.zeros_like(y_pred['class_pred'])
        anchors = y_pred['anchors']
        anchor_index_map = y_pred['anchor_index_map']
        stage_num_anchors = y_pred['stage_num_anchors']
        # ignore those pred bbox's iou with gt > ignore_iou_thresh
        bboxes_pred = y_pred['bboxes_pred'].unsqueeze(-2)
        bboxes_gt = y_label[...,:4].clone().unsqueeze(1).unsqueeze(2)
        pred_ious = BBoxBatchIOU(bboxes_pred, bboxes_gt)
        pred_ious_max = torch.max(pred_ious, dim=-1, keepdim=True)[0]
        objness_mask = pred_ious_max > self.ignore_iou_thresh
        objness_targets[objness_mask] = -1
        # assign the GT label
        anchor_index = GTBoxAnchorIndex(y_label[...,:4].clone(), anchors) 
        y_label = y_label.detach().cpu().numpy()
        anchors = anchors[0,0,...].detach().cpu().numpy()
        for b, num in enumerate(gt_bboxes_num):
            for nid in range(num):
                a_loc = anchor_index[b][nid]
                anchor = anchors[a_loc]
                a_map = anchor_index_map[a_loc]
                a_loc = int(a_loc % stage_num_anchors)
                start_index = a_map['start_index']
                width, height, stride = a_map['width'], a_map['height'], a_map['stride']
                xmin, ymin, xmax, ymax = y_label[b][nid][:4]
                b_width, b_height = xmax - xmin + 1, ymax - ymin + 1
                b_x_c, b_y_c = xmin + 0.5 * b_width, ymin + 0.5 * b_height
                b_loc_x, b_loc_y = int(b_x_c // stride), int(b_y_c // stride)
                index = start_index + b_loc_y * width + b_loc_x
                cls_id = int(y_label[b][nid][4])
                
                objness_targets[b, index, a_loc, 0] = 1
                class_targets[b, index, a_loc, cls_id] = 1
                center_targets[b, index, a_loc, 0] = b_x_c / stride - b_loc_x  
                center_targets[b, index, a_loc, 1] = b_y_c / stride - b_loc_y
                scale_targets[b, index, a_loc, 0] = np.log(b_width / anchor[0])
                scale_targets[b, index, a_loc, 1] = np.log(b_height / anchor[1])
                scale_weights[b, index, a_loc, :] = 2.0 - b_width * b_height / (width*height*stride*stride)
        return objness_targets, center_targets, scale_targets, class_targets, scale_weights
               
    def compute_loss(self, inputs):
        y_pred, y_label, gt_bboxes_num = inputs['pred'], inputs['label'], inputs['bboxes_num']
        assert len(y_label.shape) == 3 and y_label.shape[-1] == 5
        y_label[..., 4] -= 1   # ignore the background
        batch_size = y_label.shape[0]
        with torch.no_grad():
            objness_targets, center_targets, scale_targets, class_targets, scale_weights = self._target_generator(y_pred, y_label, gt_bboxes_num)
        obj_pos_mask = objness_targets > 0
        obj_neg_mask = objness_targets == 0
        box_mask = (objness_targets > 0).repeat((1,1,1,2))
        class_mask = (objness_targets > 0).repeat((1,1,1,class_targets.shape[-1]))
        raw_box_centers = y_pred['raw_box_centers']
        raw_box_scales = y_pred['raw_box_scales']
        objness_pred = y_pred['objness_pred']        
        class_pred = y_pred['class_pred']
        
        loss_xy = F.binary_cross_entropy_with_logits(input=raw_box_centers[box_mask], target=center_targets[box_mask], 
                                                     weight=scale_weights[box_mask], reduction='sum')
        loss_wh = F.mse_loss(input=raw_box_scales[box_mask], target=scale_targets[box_mask], reduction='none')
        loss_wh = (0.5 * loss_wh * scale_weights[box_mask]).sum()
        obj_pos_loss = F.binary_cross_entropy_with_logits(input=objness_pred[obj_pos_mask], target=objness_targets[obj_pos_mask],
                                                          reduction='sum')
        obj_neg_loss = F.binary_cross_entropy_with_logits(input=objness_pred[obj_neg_mask], target=objness_targets[obj_neg_mask],
                                                          reduction='sum')
        loss_obj = self.lambda_pos * obj_pos_loss + self.lambda_neg * obj_neg_loss
        loss_cls = F.binary_cross_entropy_with_logits(input=class_pred[class_mask], target=class_targets[class_mask],
                                                          reduction='sum')
        total_loss = self.weight_xy*loss_xy + self.weight_wh*loss_wh + self.weight_obj*loss_obj + self.weight_cls*loss_cls
        total_loss /= batch_size
        return {
            "loss_xy" : self.weight_xy * loss_xy / batch_size , 
            "loss_wh" : self.weight_wh * loss_wh / batch_size ,
            "loss_obj" : self.weight_obj * loss_obj / batch_size,
            "loss_cls" : self.weight_cls * loss_cls / batch_size,
            "total" : total_loss }