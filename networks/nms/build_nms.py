# encoding: utf-8
"""
@author: Chong Li
@time: 2021/06/19 11:42 morning 
@desc:
"""

from networks.nms.nms import nms

nms_dict = {
    'nms': nms,
}


def build_nms(cfg):
    _cfg = cfg.copy()
    type_name = _cfg.pop('name')
    assert type_name in nms_dict
    model = nms_dict[type_name]
    return model(**_cfg)            
