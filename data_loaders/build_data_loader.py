# encoding: utf-8
"""
@author: Chong Li
@time: 2020/06/18 11:42 morning 
@desc:
"""

from torch.utils.data import DataLoader
from data_loaders.voc import dataloader_voc
from utils.log import log_printer


loader_dict = {
    'voc': dataloader_voc.Dataset,
}


def build_data_loader(cfg, batch_size, is_train=True, is_shuffle=False, logger=None):
    _cfg = cfg.copy()
    type_name = _cfg.pop('name')
    assert type_name in loader_dict
    dataset = loader_dict[type_name](cfg, is_train, logger)
    log_printer(logger, 'total {} {} images'.format(len(dataset), 'train' if is_train else 'val'))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle, 
                            collate_fn=dataset.collate_fn, drop_last=False,
                            num_workers=cfg.get('num_workers', 0), pin_memory=True)
    return dataloader
