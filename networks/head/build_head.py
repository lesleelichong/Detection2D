# encoding: utf-8
"""
@author: Chong Li
@time: 2021/06/20 14:49
@desc:
"""

from networks.head.yolo_head import YoloHead
from networks.head.rcnn_head.rcnn_head import RcnnHead

head_dict = {
    'yolohead': YoloHead,
    'rcnnhead': RcnnHead,
}


def build_head(cfg):
    _cfg = cfg.copy()
    type_name = _cfg.pop('name')
    assert type_name in head_dict
    model = head_dict[type_name]
    return model(**_cfg)            
