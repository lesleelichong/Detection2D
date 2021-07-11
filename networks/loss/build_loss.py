# encoding: utf-8
"""
@author: Chong Li
@time: 2021/06/22 17:21
@desc:
"""

from networks.loss.yolo_loss import YoloLoss
from networks.loss.rcnn_loss import RcnnLoss

loss_dict = {
    'yololoss': YoloLoss,
    'rcnnloss': RcnnLoss,
}


def build_loss(cfg):
    _cfg = cfg.copy()
    type_name = _cfg.pop('name')
    assert type_name in loss_dict
    model = loss_dict[type_name]
    return model(**_cfg)            
