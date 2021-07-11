# encoding: utf-8
"""
@author: Chong Li
@time: 2021/06/19 11:42 morning 
@desc:
"""

from networks.neck.yolo_neck import YoloNeck
from networks.neck.fpn_neck import FeaturePyramidNetwork

neck_dict = {
    'yoloneck': YoloNeck,
    'fpn': FeaturePyramidNetwork
}


def build_neck(cfg):
    _cfg = cfg.copy()
    type_name = _cfg.pop('name')
    assert type_name in neck_dict
    model = neck_dict[type_name]
    return model(**_cfg)            
