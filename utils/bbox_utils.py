# encoding: utf-8
"""
@author: Chong Li
@time: 2021/05/27 18:39
@desc:
"""

from os import minor
import torch


def BBoxBatchIOU(boxes_a, boxes_b):
    r"""
    Args:
        tl: top left
        br: bottom right
        boxes_a, boxes_b: torch.tensor format.
    """
    tl = torch.max(boxes_a[...,:2], boxes_b[...,:2])
    br = torch.min(boxes_a[...,2:], boxes_b[...,2:])
    area_a = torch.prod(boxes_a[..., 2:] - boxes_a[..., :2] + 1, dim=-1)
    area_b = torch.prod(boxes_b[..., 2:] - boxes_b[..., :2] + 1, dim=-1)
    en = (tl < br).type(tl.type()).prod(dim=-1)
    area_i = torch.prod(br - tl + 1, dim=-1) * en
    ious =  area_i / (area_a + area_b - area_i)
    return ious


def BBoxCornerToCenter(bboxes):
    r""" (xmin,ymin,xmax,ymax) to (x_center, y_center, width, height)
    """
    assert bboxes.shape[-1] == 4
    c_bboxes = torch.zeros_like(bboxes)
    c_bboxes[..., 0] = (bboxes[..., 0] + bboxes[..., 2]) / 2
    c_bboxes[..., 1] = (bboxes[..., 1] + bboxes[..., 3]) / 2
    c_bboxes[..., 2] = (bboxes[..., 2] - bboxes[..., 0])
    c_bboxes[..., 3] = (bboxes[..., 3] - bboxes[..., 1])
    return c_bboxes


def BBoxClipToImageSize(bboxes, height, width):
    bboxes[...,0] = bboxes[...,0].clamp(min=0, max=width-1)
    bboxes[...,1] = bboxes[...,1].clamp(min=0, max=height-1)
    bboxes[...,2] = bboxes[...,2].clamp(min=0, max=width-1)
    bboxes[...,3] = bboxes[...,3].clamp(min=0, max=height-1)
    return bboxes


def BBoxSizeFilter(bboxes, scores, min_size=100):
    ws = bboxes[..., 2] - bboxes[..., 0]
    hs = bboxes[..., 3] - bboxes[..., 1]
    areas = ws * hs
    bboxes[ws <= 0] = 0
    scores[ws <= 0] = -1.0
    bboxes[hs <= 0] = 0
    scores[hs <= 0] = -1.0
    bboxes[areas < min_size] = 0
    scores[areas < min_size] = -1.0
    return bboxes, scores
