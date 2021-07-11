# encoding: utf-8
"""
@author: Chong Li
@time: 2021/06/14 10:21
@desc:
"""

import os
import argparse
import yaml
import shutil      
import numpy as np
from timeit import default_timer as timer
import torch
from torch import nn
from networks.detector import Detector
from networks.loss.build_loss import build_loss
from data_loaders.build_data_loader import build_data_loader

from utils.log import Logger, log_printer
from utils.common_utils import (get_device, resume_weights, freeze_backbone_weights,
                                convert_multi_gpu_state_dict, set_random_seed, convert_data_batch)
from utils.inference_utils import return_class_names
from metrics.metric_ap import ApMetric


class ModelTrainer(object):
    r"""This is 2D Detection Trainer Class.
    Arguments:
        one_stage_model: if True, this is yolov3, False, FPN-Faster-Rcnn
    """
    def __init__(self, cfg):
        self.one_stage_model = cfg['one_stage_model']
        self.class_num = cfg['class_num']
        self.input_channel = cfg['input_channels']
        self.save_dir = cfg['save_dir']
        cfg_train = cfg['train']
        self.train_batch_size = cfg_train['train_batch_size']
        self.val_batch_size = cfg_train['val_batch_size']
        self.max_epoch = cfg_train['max_epoch']
        self.lr_decay_epoch = cfg_train['lr_decay_epoch']
        self.lr_decay_scale = cfg_train['lr_decay_scale']
        self.first_eval_epoch = cfg_train['first_eval_epoch']
        self.eval_frequency = cfg_train['eval_frequency']
        self.log_frequency = cfg_train['log_frequency']
        self.learning_rate = cfg_train['optimizer']['lr']
        self.logger = Logger(log_file=cfg['logger_file'])
        self.device, self.device_ids = get_device(cfg['device'], self.logger)
        
        # ap metric for evaluate
        self.include_bg = cfg_train.get('include_bg', True)
        self.class_names = return_class_names(cfg['dataset']['name'])
        if self.include_bg == False:
            self.class_names = self.class_names[1:]
        metric_iou_thresh = cfg_train.get('metric_iou_thresh', 0.5)
        self.ap_metrics = ApMetric(iou_thresh=metric_iou_thresh, class_names=self.class_names)
        if self.one_stage_model == False:
            self.rpn_metrics = ApMetric(iou_thresh=metric_iou_thresh, class_names=self.class_names)
            self.rcnn_metrics = ApMetric(iou_thresh=metric_iou_thresh, class_names=self.class_names)
        # Reduce Randomness for Reproducible Result.
        set_random_seed(cfg['seed'])
        self.train_loader, self.val_loader = self._build_train_val_data_loader(cfg['dataset'])
        self.model = self._build_model(cfg['network'], cfg_train.get('resume_from', None),
                                       ignore_keys=cfg_train.get('ignore_keys', []), 
                                       freeze_keys=cfg_train.get('freeze_keys', []))
        self.optimizer = self._build_optimizer(cfg_train['optimizer'])
        self.loss_module = self._build_loss_module(cfg_train['loss'])
        self.epoch = 0
        self.map = self.best_map = 0.0
           
    def _build_train_val_data_loader(self, data_cfg):
        r"""Build Data Loader from dataset configs. """
        train_loader = build_data_loader(data_cfg, self.train_batch_size, True, True, self.logger)
        val_loader = build_data_loader(data_cfg, self.val_batch_size, False, False, self.logger)
        return train_loader, val_loader
    
    def _build_model(self, model_cfg, resume_file=None, ignore_keys=[], freeze_keys=[]):
        r"""Build network from model configs. """
        model = Detector(**model_cfg)
        model = resume_weights(model, resume_file, self.logger, ignore_keys)
        model = freeze_backbone_weights(model, self.logger, freeze_keys)
        model.to(device=self.device)
        if len(self.device_ids) > 1:
            model = nn.parallel.DataParallel(model, device_ids=self.device_ids)
        log_printer(self.logger, 'Builded {} network model'.format(model_cfg['name']))
        return model
    
    def _build_optimizer(self, optimizer_cfg):
        r"""Build optimizer from optimizer configs."""
        if len(self.device_ids) > 1:
            model = self.model.module
        else:
            model = self.model
        if optimizer_cfg['name'] == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=optimizer_cfg['lr'],
                weight_decay=optimizer_cfg['weight_decay'],
                amsgrad=optimizer_cfg['amsgrad'])
        elif optimizer_cfg['name'] == "Adam_mult_lr":
            backbone_params = list(map(id, model.backbone.parameters()))
            other_params = filter(lambda p: id(p) not in backbone_params, model.parameters())
            optimizer = torch.optim.Adam([
                {'params': model.backbone.parameters(), 'lr': optimizer_cfg['lr'] * 0.1},
                {'params': other_params, 'lr': optimizer_cfg['lr']}], 
                weight_decay=optimizer_cfg['weight_decay'],
                amsgrad=optimizer_cfg['amsgrad'])
        elif optimizer_cfg['name'] == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=optimizer_cfg['lr'],
                weight_decay=optimizer_cfg['weight_decay'],
                momentum=optimizer_cfg['momentum'])
        elif optimizer_cfg['name'] == 'SGD_mult_lr':
            backbone_params = list(map(id, model.backbone.parameters()))
            other_params = filter(lambda p: id(p) not in backbone_params, model.parameters())
            optimizer = torch.optim.SGD([
                {'params': model.backbone.parameters(), 'lr': optimizer_cfg['lr'] * 0.1},
                {'params': other_params, 'lr': optimizer_cfg['lr']}], 
                weight_decay=optimizer_cfg['weight_decay'],
                momentum=optimizer_cfg['momentum'])
        else:
            raise NotImplementedError
        log_printer(self.logger, 'Builded {} network optimizer'.format(optimizer_cfg['name']))
        return optimizer
    
    def _build_loss_module(self, loss_config):
        r"""Build loss module from loss configs."""
        return build_loss(loss_config)
    
    def _print_log_info(self, batch_id, loss_dict, speed, total_batches, mode='Train'):
        if batch_id % self.log_frequency == 0 :
            loss_dict_str = '{:.5f} ['.format(loss_dict['total'].item())
            for key in loss_dict:
                if key != 'total':
                    loss_dict_str += key + ': ' + '{:.5f}'.format(loss_dict[key].item()) + ', '
            loss_dict_str = loss_dict_str[:-2] + ']'
            message = '[{}] Epoch: {}, learning rate: {}, IterBatch: {}/{}, speed: {:.1f} imgs/s, LossBatch: {}'.format(mode,
                                                                                self.epoch, 
                                                                                str(self.learning_rate),
                                                                                batch_id, 
                                                                                total_batches,
                                                                                speed, 
                                                                                loss_dict_str)
            log_printer(self.logger, message)
            
    def _save_checkpoints(self):
        checkpoint_dict = {
            'epoch': self.epoch,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optimizer.state_dict(),
            'model_state_dict': convert_multi_gpu_state_dict(self.model.state_dict()),
            'map': self.map }
        
        save_file = os.path.join(self.save_dir, 'checkpoints', 'checkpoint_epoch_{}.pth.tar'.format(self.epoch))
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        torch.save(checkpoint_dict, save_file)
        log_printer(self.logger, 'checkpoint saved to {}'.format(save_file))
        if self.map > self.best_map:
            self.best_map = self.map
            save_file = os.path.join(self.save_dir, 'checkpoints', 'checkpoint_best_map.pth.tar')
            torch.save(checkpoint_dict, save_file)
            log_printer(self.logger, 'best map checkpoint saved to {}'.format(save_file))
        
    def _update_learning_rate(self):
        # update learning rate
        if self.epoch in self.lr_decay_epoch:
            self.learning_rate *= self.lr_decay_scale
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.lr_decay_scale
    
    def train(self):
        r"""train model pipeline """
        log_printer(self.logger, "Start training ...")
        while self.epoch < self.max_epoch:
            self._update_learning_rate()           
            self.train_epoch()
            self.epoch += 1
            if self.epoch >= self.first_eval_epoch and self.epoch % self.eval_frequency == 0:
                self.evaluate()
                self._save_checkpoints()

    def train_epoch(self):
        self.model.train()
        for batch_id, data_batch in enumerate(self.train_loader):
            t_s = timer()
            self.optimizer.zero_grad()
            inputs = convert_data_batch(data_batch, self.device)
            y_pred = self.model(inputs, is_training=True)
            inputs['pred'] = y_pred
            loss_dict = self.loss_module.compute_loss(inputs)
            loss_dict['total'].backward()
            #clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=35, norm_type=2)
            self.optimizer.step()
            speed = inputs['image'].shape[0] / (timer() - t_s)
            self._print_log_info(batch_id+1, loss_dict, speed, len(self.train_loader))
            
    def evaluate(self):
        def _run_model(inputs):
            with torch.no_grad():
                y_pred = self.model(inputs, is_training=False)
            return y_pred
        def _parse_label(inputs):
            label_list = []
            y_label = inputs['label'].detach().cpu().numpy()
            gt_bboxes_num = inputs['bboxes_num']
            for label, num in zip(y_label, gt_bboxes_num):
                if self.include_bg == False:
                    label[..., -1] -= 1
                label_list.append(label[:num, :])
            return label_list
        
        log_printer(self.logger, "Running evaluation ...")
        self.model.eval()
        for batch_id, data_batch in enumerate(self.val_loader):
            if batch_id > 9:
                break
            t_s = timer()
            inputs = convert_data_batch(data_batch, self.device)
            y_label = _parse_label(inputs)
            y_pred = _run_model(inputs)
            for one_pred, one_label in zip(y_pred['res'], y_label):
                self.ap_metrics.update(one_pred, one_label)
            if self.one_stage_model == False:
                for one_rpn, one_rcnn, one_label in zip(y_pred['rpn'],y_pred['rcnn'], y_label):
                    one_label[:,-1] = 0
                    one_rpn[..., -2] = 0
                    one_rcnn[..., -2] = 0
                    self.rpn_metrics.update(one_rpn, one_label)
                    self.rcnn_metrics.update(one_rcnn, one_label)
            message = '[Eval] Epoch: {}, learning rate: {}, IterBatch: {}/{}, speed: {:.1f} imgs/s'.format(self.epoch, 
                                                                            str(self.learning_rate), batch_id + 1, 
                                                                            len(self.val_loader), 
                                                                            inputs['image'].shape[0] / (timer() - t_s))
            log_printer(self.logger, message)
        self.map = self.ap_metrics.compute_ap_metrics(epoch=self.epoch, logger=self.logger)
        self.ap_metrics.reset()
        if self.one_stage_model == False:
            self.rpn_metrics.compute_recall_prec_metrics('rpn')
            self.rcnn_metrics.compute_recall_prec_metrics('rcnn')
            self.rpn_metrics.reset()
            self.rcnn_metrics.reset()
        

def get_cfg_from_args(args):
    cfg_file = args.cfg
    with open(cfg_file,'rt') as f:
        cfg = yaml.full_load(f)
    if args.save_dir is not None:
        save_dir = args.save_dir
    elif cfg.get('save_dir') is not None:
        save_dir = cfg['save_dir']
    else:
        raise NotImplementedError
    cfg['save_dir'] = save_dir
    logger_file = os.path.join(save_dir, 'log.txt')
    cfg['logger_file'] = logger_file
    os.makedirs(save_dir, exist_ok=True)
    shutil.copyfile(cfg_file, os.path.join(save_dir, 'config.yaml'))
    return cfg


def main(args):
    cfg = get_cfg_from_args(args)
    trainer = ModelTrainer(cfg)
    if args.usage == 'train':
        trainer.train()
    else: 
        trainer.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation Training Pipeline')
    parser.add_argument("--cfg", help="path to config file", type=str, default='configure/train/wall/wall_unet.yaml')
    parser.add_argument('--usage', type=str, default='train', help='train/val')
    parser.add_argument("--save-dir", help="output save dir", type=str)
    opts = parser.parse_args()
    cfg_list = [
        #'configure/train_local/voc/rcnn_s16.yaml'
        'configure/train_local/voc/fpn_rcnn.yaml'
    ]
    for cfg_file in cfg_list:
        opts.cfg = cfg_file
        main(opts)
