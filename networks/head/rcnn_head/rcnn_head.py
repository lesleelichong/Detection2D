# encoding: utf-8
"""
@author: Chong Li
@time: 2021/06/28 18:39
@desc:
"""

import numpy as np
import torch
import torchvision
from torch import nn
from networks.head.rcnn_head.rcnn_utils import MultiScaleRoIAlign, TwoMLPHead, FastRCNNPredictor
from utils.bbox_utils import BBoxClipToImageSize, BBoxSizeFilter, BBoxBatchIOU
from networks.rpn.rpn_utils import (Matcher, BalancedPositiveNegativeSampler, 
                                    NormalizedProposalEncoder, NormalizedProposalDecoder)


class RcnnHead(nn.Module):
    def __init__(self, 
                 roi_pool_size=7, roi_in_channels=256,
                 roi_out_channels=1024, 
                 num_classes=20,
                 include_bg=False,
                 score_thresh=0.1,
                 nms_thresh=0.5,
                 proposal_stds=(0.1, 0.1, 0.2, 0.2), 
                 proposal_means=(0., 0., 0., 0.),
                 fg_iou_thresh=0.5, 
                 bg_iou_thresh=0.5, 
                 num_sample=512, 
                 pos_ratio=0.25 ):
        super(RcnnHead, self).__init__()
        # cls and box regress all include the bg
        if include_bg == False:
            num_classes += 1
        self.include_bg = include_bg
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        bg_iou_thresh = min(bg_iou_thresh, fg_iou_thresh)
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_sample, pos_ratio)
        self.proposal_encode = NormalizedProposalEncoder(stds=proposal_stds, means=proposal_means)
        self.proposal_decode = NormalizedProposalDecoder(stds=proposal_stds, means=proposal_means)
        self.roi_pool = MultiScaleRoIAlign(output_size=roi_pool_size)
        self.roi_head = TwoMLPHead(roi_in_channels*roi_pool_size**2, roi_out_channels)
        self.roi_predict = FastRCNNPredictor(roi_out_channels, num_classes)
        
    def incoming_label(self, inputs):
        self.gt_label = inputs['label'].clone()
        self.gt_bboxes_num = inputs['bboxes_num']
        
    def forward(self, inputs, is_training=False):
        if is_training == True:
            return self._forward_train(inputs)
        else:
            return self._forward_test(inputs)
    
    def _forward_test(self, inputs):
        r"""
        rpn_props, rcnn_props: we use the rpn/rcnn intermediate proposals to
        analyze the more detailed performance of the detector
        """
        proposals = inputs['proposals']
        props_feature = self.roi_pool(inputs['features'], inputs['proposals'], 
                                      inputs['anchor_sizes'], inputs['strides'])
        
        props_feature = self.roi_head(props_feature)
        res = self.roi_predict(props_feature)
        bboxes_score = res['props_score']
        bboxes_pred = res['props']
        num_classes = bboxes_score.shape[-1]
        image_h = inputs['features'][0].shape[-2] * inputs['strides'][0]
        image_w = inputs['features'][0].shape[-1] * inputs['strides'][0]
        
        bboxes_per_image = [len(b) for b in proposals]
        proposals = torch.cat(proposals, dim=0).unsqueeze(1)
        bboxes_score = torch.softmax(bboxes_score, dim=-1)
        bboxes_pred = bboxes_pred.reshape(len(proposals), -1, 4)
        bboxes_pred = self.proposal_decode(bboxes_pred, proposals)
        bboxes_score = bboxes_score.split(bboxes_per_image, 0)
        bboxes_pred = bboxes_pred.split(bboxes_per_image, 0)
        proposals = proposals.squeeze(1).split(bboxes_per_image, 0)
        
        keep_res = []
        rpn_props = []
        rcnn_props = []
        for bboxes, scores, props in zip(bboxes_pred, bboxes_score, proposals):
            # create labels for each prediction
            labels = torch.arange(num_classes, device=scores.device)
            labels = labels.view(1, -1).expand_as(scores)
            # remove predictions with the background label
            bboxes, scores, labels = bboxes[:, 1:], scores[:, 1:], labels[:, 1:]
            max_scores = torch.max(scores, dim=-1, keepdim=True)[0]
            mask = scores == max_scores
            bboxes, scores, labels = bboxes[mask], scores[mask], labels[mask]
            #bboxes, scores, labels = bboxes.reshape(-1, 4), scores.reshape(-1), labels.reshape(-1)
            bboxes = BBoxClipToImageSize(bboxes, image_h, image_w)
            bboxes, scores = BBoxSizeFilter(bboxes, scores, 0)
            rcnn_bboxes = bboxes.clone().detach().cpu().numpy()
            props = props.detach().cpu().numpy()
            rpn_props.append(np.pad(props, ((0,0), (0,2))))
            rcnn_props.append(np.pad(rcnn_bboxes, ((0,0), (0,2))))
            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            if len(inds) ==0:
                keep_res.append(np.array([]).reshape(-1,6))
                continue
            bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
            # non-maximum suppression, independently done per class
            keep = torchvision.ops.batched_nms(bboxes, scores, labels, self.nms_thresh)
            bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]
            if self.include_bg == False:
                labels -= 1
            # (N, [xmin，ymin,xmax,ymax, class, score])
            res = torch.cat([bboxes, labels.unsqueeze(-1), scores.unsqueeze(-1)], dim=-1)
            keep_res.append(res.detach().cpu().numpy())
        return {
            'res': keep_res,
            'rpn': rpn_props,
            'rcnn': rcnn_props
        }
    
    def _forward_train(self, inputs):
        props_info = self._generate_train_props_info(inputs['proposals'])
        props_feature = self.roi_pool(inputs['features'], props_info['proposals'], 
                                      inputs['anchor_sizes'], inputs['strides'])
        props_feature = self.roi_head(props_feature)
        res = self.roi_predict(props_feature)
        res['props_label'] = torch.cat(props_info['props_label'], dim=0)
        res['props_targets'] = torch.cat(props_info['props_targets'], dim=0)
        res['rpn'] = inputs['training']
        return res
         
    def _sample_props_idx(self, cls_label):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(cls_label)
        sampled_inds = []
        for pos_inds_img, neg_inds_img in zip(sampled_pos_inds, sampled_neg_inds):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def _encode_proposals(self, reference_boxes, proposals):
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.proposal_encode(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)
    
    def _assign_targets_to_proposals(self, proposals):
        cls_label_list = []
        matched_idxs_list = []
        for b_proposals, b_label, gt_num in zip(proposals, self.gt_label, self.gt_bboxes_num):
            if gt_num == 0:
                cls_label = torch.zeros(len(b_proposals), dtype=b_proposals.dtype, 
                                            device=b_proposals.device)
                matched_idxs = torch.zeros(len(b_proposals), dtype=b_proposals.dtype, 
                                            device=b_proposals.device) - 1
                cls_label_list.append(cls_label)
                matched_idxs_list.append(matched_idxs)
                continue
            gt_bboxes = b_label[:gt_num][...,:4]
            gt_cls_label = b_label[:gt_num][...,4]
            ious = BBoxBatchIOU(gt_bboxes.unsqueeze(1), b_proposals.unsqueeze(0))
            matched_idxs = self.proposal_matcher(ious)
            cls_label = gt_cls_label[matched_idxs.clamp(min=0)]
            # Label background (below the low threshold)
            bg_inds = matched_idxs==self.proposal_matcher.BELOW_LOW_THRESHOLD
            cls_label[bg_inds] = 0
            matched_idxs[bg_inds] = -1
            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
            cls_label[ignore_inds] = -1
            matched_idxs[ignore_inds] = -1
            cls_label_list.append(cls_label)
            matched_idxs_list.append(matched_idxs)
        return cls_label_list, matched_idxs_list
  
    def _generate_train_props_info(self, proposals):
        # append ground-truth bboxes to propos
        proposals = [
            torch.cat((props, b_label[:gt_num][...,:4]))
            for props, b_label, gt_num in zip(proposals, self.gt_label, self.gt_bboxes_num)]
        matched_cls_label, matched_idxs = self._assign_targets_to_proposals(proposals)
        sampled_inds = self._sample_props_idx(matched_cls_label)
        matched_gt_bboxes = []
        for idx, (sam_inds, b_label, gt_num) in enumerate(zip(sampled_inds, self.gt_label, self.gt_bboxes_num)):
            proposals[idx] = proposals[idx][sam_inds]
            matched_cls_label[idx] = matched_cls_label[idx][sam_inds]
            matched_idxs[idx] = matched_idxs[idx][sam_inds]
            gt_bboxes = b_label[:gt_num][...,:4]
            if gt_num == 0:
                gt_bboxes = torch.zeros((1, 4), dtype=proposals[idx].dtype, device=proposals[idx].device)
            matched_bboxes = gt_bboxes[matched_idxs[idx].clamp(min=0)]
            # Background (negative examples)
            bg_inds = matched_idxs[idx] < 0
            matched_bboxes[bg_inds] = 0
            matched_gt_bboxes.append(matched_bboxes)
        proposals_targets = self._encode_proposals(matched_gt_bboxes, proposals)
        return {
            'proposals': proposals,
            'props_label': matched_cls_label,
            'props_targets': proposals_targets }