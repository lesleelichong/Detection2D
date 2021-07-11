# encoding: utf-8
"""
@author: Chong Li
@time: 2021/06/27 16:28
@desc:
"""

from networks.rpn.rpn import RPN

rpn_dict = {
    'rpn': RPN,
}


def build_rpn(cfg):
    _cfg = cfg.copy()
    type_name = _cfg.pop('name')
    assert type_name in rpn_dict
    model = rpn_dict[type_name]
    return model(**_cfg)            
