# encoding: utf-8
"""
@author: Chong Li
@time: 2021/07/01 9:39
@desc:
"""


import numpy as np
import torch
from torch import nn
from torchvision.ops import roi_align


class MultiScaleRoIAlign(nn.Module):
    """
    Multi-scale RoIAlign pooling, which is useful for detection with or without FPN.
    It infers the scale of the pooling via the heuristics present in the FPN paper.
    Arguments:
        output_size (List[Tuple[int, int]] or List[int]): output size for the pooled region
        sampling_ratio (int): sampling ratio for ROIAlign
    """
    def __init__(self, output_size, sampling_ratio=2):
        super(MultiScaleRoIAlign, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)

    def _convert_to_roi_format(self, bboxes):
        concat_bboxes = torch.cat(bboxes, dim=0)
        device, dtype = concat_bboxes.device, concat_bboxes.dtype
        ids = torch.cat([torch.full_like(b[:, :1], i, dtype=dtype, layout=torch.strided, device=device)
                         for i, b in enumerate(bboxes)], dim=0)
        rois = torch.cat([ids, concat_bboxes], dim=-1)
        return rois
    
    def _generate_split_areas(self, anchor_sizes):
        split_areas = []
        areas = [np.inf]
        for i in range(len(anchor_sizes)-1):
            s = min(anchor_sizes[i]) + max(anchor_sizes[i+1])
            s = s/2
            areas.append(s*s)
        areas.append(-np.inf)
        for i in range(len(areas) - 1):
            split_areas.append([areas[i+1], areas[i]])
        return split_areas
    
    def forward(self, features, bboxes, anchor_sizes, strides):
        r"""
        anchor_sizes sort from large to small
        """
        assert len(features) == len(anchor_sizes) == len(strides)
        scales = [1./s for s in strides]
        rois = self._convert_to_roi_format(bboxes)
        if len(features) == 1:
            return roi_align(
                features[0], rois,
                output_size=self.output_size,
                spatial_scale=scales[0],
                sampling_ratio=self.sampling_ratio)
        else:
            split_areas = self._generate_split_areas(anchor_sizes)
            areas = (rois[...,4] - rois[...,2]) * (rois[...,3] - rois[...,1])
            levels = torch.zeros_like(areas, dtype=torch.int64)
            for idx, (s_min, s_max) in enumerate(split_areas):
                levels[(areas>=s_min) & (areas < s_max)] = idx
            num_rois = len(rois)
            num_channels = features[0].shape[1]
            result = torch.zeros((num_rois, num_channels,) + self.output_size,
                                  dtype=features[0].dtype, device=features[0].device)
            for level, (per_level_feature, scale) in enumerate(zip(features, scales)):
                idx_in_level = torch.where(levels == level)[0]
                rois_per_level = rois[idx_in_level]
                result_idx_in_level = roi_align(
                    per_level_feature, rois_per_level,
                    output_size=self.output_size,
                    spatial_scale=scale, sampling_ratio=self.sampling_ratio)
                result[idx_in_level] = result_idx_in_level.to(result.dtype)   
            return result


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models
    Arguments:
        in_channels (int): number of input channels
        out_channels (int): size of the intermediate representation
    """
    def __init__(self, in_channels, out_channels):
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, out_channels)
        self.fc7 = nn.Linear(out_channels, out_channels)
        self.crit = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.crit(self.fc6(x))
        x = self.crit(self.fc7(x))
        return x
    
class FastRCNNPredictor(nn.Module):
    r"""
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        bbox_scores = self.cls_score(x)
        bbox_preds = self.bbox_pred(x)
        return {
            'props': bbox_preds,
            'props_score': bbox_scores }
