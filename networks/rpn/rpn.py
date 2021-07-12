# encoding: utf-8
"""
@author: Chong Li
@time: 2021/05/27 18:39
@desc:
"""

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from networks.rpn.rpn_utils import generate_anchors, get_batch_scores_top_n_idx
from networks.rpn.rpn_utils import (NormalizedProposalDecoder, Matcher, 
                                    BalancedPositiveNegativeSampler, NormalizedProposalEncoder)
from utils.bbox_utils import BBoxClipToImageSize, BBoxSizeFilter, BBoxBatchIOU
from networks.rpn.rpn_head import rpn_head


class RPN(nn.Module):
    r"""Region Proposal Network.
    TODO:
        1. Although rpn_mode_type has softmax Interface. Now rpn objness classification only support sigmoid. 
        2. The Training/Inference Parameters are confused together. Maybe there is an Base Class to solve it.
        3. rpn_head_type: Now only support StandardRPNHead.
    """
    def __init__(self, in_channels, anchor_sizes, aspect_ratios, strides, 
                 rpn_head_type, rpn_mode_type='sigmoid',
                 topk_pre_nms_proposals_test=2000,
                 topk_post_nms_proposals_test=300,
                 nms_thresh=0.7,
                 proposal_min_area=[25, 100, 400],
                 proposal_stds=(1., 1., 1., 1.), 
                 proposal_means=(0., 0., 0., 0.),
                 topk_pre_nms_proposals_train=6000, 
                 topk_post_nms_proposals_train=2000, 
                 fg_iou_thresh=0.7, 
                 bg_iou_thresh=0.3,
                 num_sample=256,
                 pos_ratio=0.5,
                 alloc_size=(128, 128)):
        super(RPN, self).__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels]
        if isinstance(proposal_min_area, int):
            proposal_min_area = [proposal_min_area for _ in in_channels]
        for in_ch in in_channels:
            assert in_ch == in_channels[0]
        anchor_sizes = anchor_sizes[::-1][:len(in_channels)]
        strides = strides[::-1][:len(in_channels)]
        
        self.topk_pre_nms_proposals_train = topk_pre_nms_proposals_train
        self.topk_pre_nms_proposals_test = topk_pre_nms_proposals_test
        self.topk_post_nms_proposals_train = topk_post_nms_proposals_train
        self.topk_post_nms_proposals_test = topk_post_nms_proposals_test
        self.proposal_min_area = proposal_min_area[::-1]
        self.nms_thresh = nms_thresh
        self.strides = strides
        self.anchor_sizes = anchor_sizes
        # Generate 128*128 size anchors map ahead of time
        self.anchors = generate_anchors(strides, anchor_sizes, 
                                        aspect_ratios, alloc_size)
        self.rpn_head = rpn_head(rpn_head_type, in_channels[0], 
                                 len(anchor_sizes[0]) * len(aspect_ratios), rpn_mode_type)
        self.proposal_encode = NormalizedProposalEncoder(stds=proposal_stds, means=proposal_means)
        self.proposal_decode = NormalizedProposalDecoder(stds=proposal_stds, means=proposal_means)
        self.anchor_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_sample, pos_ratio)

    def incoming_label(self, inputs):
        self.gt_label = inputs['label']
        self.gt_bboxes_num = inputs['bboxes_num']
        
    def _sample_props_idx(self, cls_label):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(cls_label)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        return sampled_pos_inds, sampled_inds
    
    def _assign_targets_to_anchors(self, anchors):
        cls_label_list = []
        matched_gt_bboxes_list = []
        for b_label, gt_num in zip(self.gt_label, self.gt_bboxes_num):
            if gt_num == 0:
                cls_label = torch.zeros(anchors.shape[1], dtype=b_label.dtype, 
                                         device=b_label.device)
                matched_gt_bboxes = torch.zeros((anchors.shape[1], 4), dtype=b_label.dtype, 
                                                 device=b_label.device)
                cls_label_list.append(cls_label)
                matched_gt_bboxes_list.append(matched_gt_bboxes)
                continue
            gt_bboxes = b_label[:gt_num][...,:4]
            ious = BBoxBatchIOU(gt_bboxes.unsqueeze(1), anchors)
            matched_idxs = self.anchor_matcher(ious)
            matched_gt_bboxes = gt_bboxes[matched_idxs.clamp(min=0)]
            cls_label = matched_idxs >= 0
            cls_label = cls_label.to(dtype=torch.float32)
            # Background (negative examples)
            bg_inds = matched_idxs==self.anchor_matcher.BELOW_LOW_THRESHOLD
            cls_label[bg_inds] = 0
            matched_gt_bboxes[bg_inds] = 0
            # discard indices that are between thresholds
            ignore_inds = matched_idxs == self.anchor_matcher.BETWEEN_THRESHOLDS
            cls_label[ignore_inds] = -1
            matched_gt_bboxes[ignore_inds] = 0
            cls_label_list.append(cls_label)
            matched_gt_bboxes_list.append(matched_gt_bboxes)
        return cls_label_list, matched_gt_bboxes_list
        
    def _concat_stages_prediction(self, layers):
        scores_list = []
        proposals_list = []
        rpn_raw_scores_list = []
        rpn_raw_bboxes_list = []
        anchors_list = []
        num_anchors_per_stage = []
        rpn_scores, rpn_bboxes, rpn_raw_scores, rpn_raw_bboxes = self.rpn_head(layers)
        for (scores, bboxes, raw_scores, raw_bboxes, anchors, stride, min_area) in zip(rpn_scores, rpn_bboxes, 
                                                                             rpn_raw_scores, rpn_raw_bboxes, self.anchors,
                                                                             self.strides, self.proposal_min_area):
            b, h, w, num = bboxes.shape[:4]
            anchors = torch.from_numpy(anchors[:h, :w, ...]).unsqueeze(0).to(bboxes.device)
            proposals = self.proposal_decode(bboxes, anchors)
            proposals = BBoxClipToImageSize(proposals, h*stride, w*stride)
            proposals, scores = BBoxSizeFilter(proposals, scores, min_area)
            num_anchors_per_stage.append(h*w*num)
            anchors_list.append(anchors.reshape(-1, 4))
            rpn_raw_scores_list.append(raw_scores.reshape(b, -1))
            rpn_raw_bboxes_list.append(raw_bboxes.reshape(b, -1, 4))
            scores_list.append(scores.reshape(b, -1))
            proposals_list.append(proposals.reshape(b, -1, 4))
        anchors = torch.cat(anchors_list, dim=0)
        rpn_raw_scores = torch.cat(rpn_raw_scores_list, dim=1).reshape(-1)
        rpn_raw_bboxes = torch.cat(rpn_raw_bboxes_list, dim=1).reshape(-1, 4)
        proposals = torch.cat(proposals_list, dim=1)
        scores = torch.cat(scores_list, dim=1)
        return proposals, scores, anchors, num_anchors_per_stage, rpn_raw_bboxes, rpn_raw_scores
            
    def _select_proposals(self, proposals, scores, num_anchors_per_stage, 
                          topk_pre_nms_proposals, topk_post_nms_proposals):
        top_k_idx = get_batch_scores_top_n_idx(scores, num_anchors_per_stage, topk_pre_nms_proposals)
        levels = torch.cat([
            torch.full((n,), idx, dtype=torch.int64, device=scores.device)
            for idx, n in enumerate(num_anchors_per_stage)], 0)
        levels = levels.reshape(1, -1).expand_as(scores)
        image_range = torch.arange(len(scores), device=scores.device)
        batch_idx = image_range[:, None]
        scores = scores[batch_idx, top_k_idx]
        levels = levels[batch_idx, top_k_idx]
        proposals = proposals[batch_idx, top_k_idx]
        keep_scores = []
        keep_proposals = []
        for b_proposals, b_scores, cls_idx in zip(proposals, scores, levels):
            keep = b_scores >= 0
            b_proposals, b_scores, cls_idx = b_proposals[keep], b_scores[keep], cls_idx[keep]
            # non-maximum suppression, independently done per level
            keep = torchvision.ops.batched_nms(b_proposals, b_scores, cls_idx, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:topk_post_nms_proposals] 
            keep_proposals.append(b_proposals[keep])
            keep_scores.append(b_scores[keep])
        return keep_proposals, keep_scores

    def forward(self, layers, is_training=False):
        if is_training == True:
            return self._forward_train(layers)
        else:
            return self._forward_test(layers)
        
    def _forward_test(self, layers):
        scores_list = []
        proposals_list = []
        num_anchors_per_stage = []
        rpn_scores, rpn_bboxes = self.rpn_head(layers)[:2]
        for (scores, bboxes, anchors, stride, min_area) in zip(rpn_scores, rpn_bboxes, self.anchors,
                                                               self.strides, self.proposal_min_area):
            b, h, w, num = bboxes.shape[:4]
            anchors = torch.from_numpy(anchors[:h, :w, ...]).unsqueeze(0).to(bboxes.device)
            proposals = self.proposal_decode(bboxes, anchors)
            proposals = BBoxClipToImageSize(proposals, h*stride, w*stride)
            proposals, scores = BBoxSizeFilter(proposals, scores, min_area)
            num_anchors_per_stage.append(h*w*num)
            scores_list.append(scores.reshape(b, -1))
            proposals_list.append(proposals.reshape(b, -1, 4))
        proposals = torch.cat(proposals_list, dim=1)
        scores = torch.cat(scores_list, dim=1)
        proposals, scores = self._select_proposals(proposals, scores, num_anchors_per_stage, 
                                                   self.topk_pre_nms_proposals_test, self.topk_post_nms_proposals_test)
        return {
                'proposals': proposals,
                'scores': scores,
                'anchor_sizes': self.anchor_sizes,
                'strides': self.strides }
        
    def _forward_train(self, layers):
        proposals, scores, anchors, num_anchors_per_stage, rpn_raw_bboxes, rpn_raw_scores = self._concat_stages_prediction(layers)
        proposals, scores = self._select_proposals(proposals, scores, num_anchors_per_stage, 
                                                   self.topk_pre_nms_proposals_train, self.topk_post_nms_proposals_train)
        cls_label, matched_gt_bboxes = self._assign_targets_to_anchors(anchors)
        sampled_pos_inds, sampled_inds = self._sample_props_idx(cls_label)
        
        batch_anchors = torch.cat([anchors for _ in range(len(cls_label))], dim=0)
        batch_cls_label = torch.cat(cls_label, dim=0)
        batch_matched_gt_bboxes = torch.cat(matched_gt_bboxes, dim=0)
        rpn_props_label = batch_cls_label[sampled_inds]
        rpn_props_score = rpn_raw_scores[sampled_inds]
        rpn_props = rpn_raw_bboxes[sampled_pos_inds]
        rpn_props_anchors = batch_anchors[sampled_pos_inds]
        rpn_props_targets = batch_matched_gt_bboxes[sampled_pos_inds]
        rpn_props_targets = self.proposal_encode(rpn_props_targets, rpn_props_anchors)
        return {
                'proposals': proposals,
                'scores': scores,
                'anchor_sizes': self.anchor_sizes,
                'strides': self.strides,
                'training': {
                    'props': rpn_props,
                    'props_targets': rpn_props_targets,
                    'props_label': rpn_props_label,
                    'props_score': rpn_props_score,}
                }
