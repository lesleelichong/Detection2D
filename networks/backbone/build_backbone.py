# encoding: utf-8
"""
@author: Chong Li
@time: 2020/12/28 11:42 morning 
@desc:
"""

from networks.backbone.resnet import *


backbone_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50, 
    'resnet101': resnet101,
    'resnext50_32x4d': resnext50_32x4d, 
    'resnext101_32x8d': resnext101_32x8d
}


def build_backbone(cfg):
    _cfg = cfg.copy()
    type_name = _cfg.pop('name')
    assert type_name in backbone_dict
    model = backbone_dict[type_name]
    return model(**_cfg)            
