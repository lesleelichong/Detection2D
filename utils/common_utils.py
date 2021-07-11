
"""
@author: Chong Li
@time: 2021/04/09 22:37
@desc:
"""

import os
import random
import torch
import numpy as np
from collections import OrderedDict
from utils.log import log_printer


def get_device(device_str='', logger=None, device_name=None):
    r"""
    Args:
        device_str: using ',' to split device names; e.g. '1,2'; default ''
    """
    device_ids = []
    if device_name != 'cpu':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = device_str
        device_ids = [int(x) for x in device_str.split(',')]
        device_ids = device_ids if len(device_ids) > 1 else [0]
        device_name = 'cuda:' + str(device_ids[0])
    device = torch.device(device_name)
     
    if logger is not None:
        if device_name != 'cpu':
            log_printer(logger, 'Let\'s use %d GPU(s)!' % len(device_ids))
        else:
            log_printer(logger, 'Let\'s use CPU!')
    return device, device_ids


def resume_weights(model, resume_file=None, logger=None, ignore_keys=[]):
    if resume_file is not None and os.path.exists(resume_file):
        message = 'load pre-trained weights from {}'.format(resume_file) 
        log_printer(logger, message)
        resume_dict = torch.load(resume_file, map_location=torch.device('cpu'))
        if 'model_state_dict' in resume_dict:
            resume_dict = resume_dict['model_state_dict']
        model_dict = model.state_dict()
        for name, param in resume_dict.items():
            if name not in model_dict or name in ignore_keys:
                log_printer(logger, 'ignore parameter named {}'.format(name))
                continue
            try:
                with torch.no_grad():
                    model_dict[name].copy_(param)
            except:
                log_printer(logger, 'fail to load parameter named {}'.format(name))
        for name, param in model_dict.items():
            if name not in resume_dict:
                log_printer(logger, 'not loaded parameter named {}'.format(name))
    return model


def freeze_backbone_weights(model, logger=None, no_freeze_keys=[]):
    if len(no_freeze_keys) == 0:
        return model
    for name, parameter in model.backbone.named_parameters():
        if any([layer in name for layer in no_freeze_keys]):
            continue
        log_printer(logger, 'freeze parameter named {}'.format(name))
        parameter.requires_grad_(False)
    return model
    

def convert_multi_gpu_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # remove `module.` of DataParallel model
        new_k = k[7:] if k.startswith('module.') else k
        new_state_dict[new_k] = v
    return new_state_dict


def convert_data_batch(data_batch, device):
    image, label, bboxes_num, scene_ids = data_batch
    image = image.to(device)
    label = label.to(device)
    return {
        'image': image,
        'label': label,
        'bboxes_num': bboxes_num,
        'scene_ids': scene_ids }

   
def set_random_seed(seed=0, deterministic=False):
    """Reduce Randomness for Reproducible Result.
    Args:
        seed (int): Seed to be used.
        deterministic (bool):
            Remove randomness (may be slower on Tesla GPUs) 
            # https://pytorch.org/docs/stable/notes/randomness.html
            Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    # torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
