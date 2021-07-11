# encoding: utf-8
"""
@author: Chong Li
@time: 2021/06/25 16:35
@desc:
"""

import torch
import numpy as np


class nms(object):
    def __init__(self, score_thresh, nms_thresh):
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
      
    def _score_filter(self, res):
        mask = res[..., -1] >= self.score_thresh
        return res[mask]
    
    def _unique_class_pool(self, res):
        return torch.unique(res[...,4])
        
    def _one_class_nms(self, res):
        keep = []
        order = torch.sort(res[..., 5], descending=True)[1]
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            keep.append(order[0].item())
            areas = (res[...,2] - res[...,0] + 1).clamp(min=0) * (res[...,3] - res[...,1] + 1).clamp(min=0)  
            xmin = res[order[1:]][..., 0].clamp(min=res[order[0]][0])
            ymin = res[order[1:]][..., 1].clamp(min=res[order[0]][1])
            xmax = res[order[1:]][..., 2].clamp(max=res[order[0]][2])
            ymax = res[order[1:]][..., 3].clamp(max=res[order[0]][3])
            inter = (xmax - xmin + 1).clamp(min=0) * (ymax - ymin + 1).clamp(min=0)
            union = areas[order[1:]] + areas[order[0]] - inter
            ious = inter / union
            idx = (ious < self.nms_thresh).nonzero(as_tuple=True)[0]
            # idx start from order[1:]
            order = order[idx+1]
        return keep
            
    def _process_one_image(self, res):
        r"""res: (N, [xmin，ymin,xmax,ymax,class, score])"""
        keep_res = []
        res = self._score_filter(res)
        class_pool = self._unique_class_pool(res)
        for cid in class_pool:
            c_res = res[res[...,4] == cid]
            keep = self._one_class_nms(c_res)
            keep_res.append(c_res[keep])
        if len(keep_res) !=0:
            keep_res = torch.cat(keep_res, dim=0).detach().cpu().numpy()
        else:
            keep_res = np.array([]).reshape(-1,6)
        return keep_res
    
    def process(self, res):
        keep_res = []
        for one_res in res:
            keep_res.append(self._process_one_image(one_res))
        return {'res': keep_res}
