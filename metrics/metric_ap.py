# encoding: utf-8
"""
@author: Chong Li
@time: 2021/06/26 15:28
@desc:
"""

from collections import defaultdict
import numpy as np
from utils.log import log_printer


class ApMetric(object):
    r"""Calculate mean AP for object detection task"""
    def __init__(self, iou_thresh=0.5, class_names=None):
        self.class_names = list(class_names)
        self.class_num = len(class_names)
        self.iou_thresh = iou_thresh
        self.reset()

    def reset(self):
        r"""Clear the internal statistics to initial state."""
        self.num_gt = [0 for _ in self.class_names]
        self.scores = [[] for _ in self.class_names]
        self.match = [[] for _ in self.class_names]

    def _cal_ious(self, pred, label):
        r"""
        Args:
            pred: (N, [xmin，ymin, xmax, ymax])
            label: (M, [xmin，ymin, xmax,ymax])
            tl: top left
            br: bottom right
            ious: (N,M)
        """
        N, M = len(pred), len(label)
        pred = np.tile(pred[:,None,:], (1,M,1))
        label = np.tile(label[None,:,:], (N,1,1))
        tl = np.maximum(pred[...,:2], label[...,:2])
        br = np.minimum(pred[...,2:], label[...,2:])
        area_a = np.prod(pred[..., 2:] - pred[..., :2] + 1, axis=-1)
        area_b = np.prod(label[..., 2:] - label[..., :2] + 1, axis=-1)
        en = (tl < br).astype(tl.dtype).prod(axis=-1)
        area_i = np.prod(br - tl + 1, axis=-1) * en
        ious =  area_i / (area_a + area_b - area_i)
        return ious
        
    def update(self, pred, label):
        r"""Update internal buffer with latest one image prediction and gt pairs.
        Args:
            pred: (N, [xmin，ymin, xmax, ymax, class, score])
            label: (M, [xmin，ymin, xmax,ymax,class])
        """
        assert len(pred.shape) == 2 and len(label.shape) == 2
        # fisrt whole sort by score
        order = np.argsort(pred[...,-1])[::-1]
        pred = pred[order]
        class_pool = np.unique(pred[...,4])
        for cid in class_pool[::-1]:
            cid_pred = pred[pred[...,4]==cid]
            cid_label = label[label[...,-1]==cid]
            cid = int(cid)
            self.scores[cid] += cid_pred[...,-1].tolist()
            if len(cid_label) == 0:
                self.match[cid] += [0]*len(cid_pred)
                continue
            ious = self._cal_ious(cid_pred[...,:4], cid_label[...,:4])
            match_gt = ious.argmax(axis=-1)
            match_gt[ious.max(axis=-1) < self.iou_thresh] = -1  
            cid_match = [0 for _ in range(len(cid_pred))]
            used_gt = np.zeros(len(cid_label), dtype=np.bool)
            for idx, gt_index in enumerate(match_gt):
                if gt_index == -1:
                    continue
                if used_gt[gt_index]:
                    continue
                cid_match[idx] = 1
                used_gt[gt_index] = True
            self.match[cid] += cid_match
        # total gt
        label_pool = np.unique(label[...,-1])
        for cid in label_pool:
            cid_label = label[label[...,-1]==cid]
            cid = int(cid)
            self.num_gt[cid] += len(cid_label)
    
    def _cal_one_class_prec_recall(self, scores, match, num_gt):
        scores = np.array(scores)
        match = np.array(match)
        order = scores.argsort()[::-1]
        match = match[order]
        tp = np.cumsum(match == 1)
        fp = np.cumsum(match == 0)
        prec = tp / (tp + fp)
        recall = tp / num_gt
        return prec, recall
    
    def _average_precision(self, prec, rec):
        r"""calculate average precision
        """
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], np.nan_to_num(prec), [0.]))
        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # look for recall value changes
        index = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[index + 1] - mrec[index]) * mpre[index + 1])
        return ap
 
    def compute_ap_metrics(self, epoch='best', dtype='validation', logger=None):
        r"""include each class AP and mAP for all classes
        """
        aps = []
        for scores, match, num_gt in zip(self.scores, self.match, self.num_gt):
            if num_gt == 0 and len(scores) == 0:
                aps.append(np.nan)
                continue
            if num_gt == 0 and len(scores) != 0:
                aps.append(0)
                continue
            prec, recall = self._cal_one_class_prec_recall(scores, match, num_gt)
            ap = self._average_precision(prec, recall)
            aps.append(ap)
        mAP = np.nanmean(aps)
        
        # print log
        print_str = 'metrics for {} epoch in {} datasets: \n'.format(str(epoch), dtype)
        for name, ap in zip(self.class_names, aps):
            if ap == np.nan:
                print_str += '{}\n AP=NULL\n'.format(name)
            else:
                print_str += '{}\n AP={:.3f}\n'.format(name, ap)
        print_str += 'Average metrics:\nmAP={:.3f}\n'.format(mAP)
        log_printer(logger, print_str)
        return mAP
    
    def compute_recall_prec_metrics(self, name, epoch='best', dtype='validation', logger=None):
        recalls = []
        precs = []
        for match, num_gt in zip(self.match, self.num_gt):
            if num_gt == 0:
                recalls.append(np.nan)
                precs.append(np.nan)
                continue 
            match = np.array(match) 
            tp = (match == 1).sum()
            fp = (match == 0).sum()   
            precs.append(tp / (tp + fp))
            recalls.append(tp / num_gt) 
        # print log
        print_str = 'recall metrics for {} epoch in {} datasets: \n'.format(str(epoch), dtype)
        print_str += '{}_objectness\n recall={:.3f}, prec={:.3f}\n'.format(name, recalls[0], precs[0])
        log_printer(logger, print_str)
        