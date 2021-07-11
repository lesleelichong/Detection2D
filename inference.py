# encoding: utf-8
"""
@author: Chong Li
@time: 2021/06/25 10:38
@desc:
"""

import os
import argparse
from typing import OrderedDict
import yaml
import shutil
import numpy as np
import cv2   
from timeit import default_timer as timer
import torch

from networks.detector import Detector
from utils.log import Logger, log_printer
from utils.common_utils import (get_device, resume_weights)
from utils.inference_utils import (parse_scene_file, return_class_names, 
                                   read_image, read_label, scale_image_and_label)
from metrics.metric_ap import ApMetric


class ModelInference(object):
    r"""This is Object Detection Inference Class.
    """
    def __init__(self, cfg, device_name=None):
        # basic parameters
        self.data_name = cfg['data_name']
        self.include_bg = cfg.get('include_bg', True)
        self.class_names = return_class_names(cfg['data_name'])
        if self.include_bg == False:
            self.class_names = self.class_names[1:]
        self.data_dir = cfg['data_dir']
        self.save_dir = cfg['save_dir']
        self.class_num = cfg['class_num']
        self.input_size = cfg['input_size']
        self.image_mean = cfg['image_mean']
        self.image_std = cfg['image_std']
        self.label_dir = cfg.get('label_dir')
        self.keep_ratio = cfg.get('keep_ratio', False)
        self.pipeline = cfg.get('pipeline', OrderedDict())
        self.data_kwargs = cfg.get('data_kwargs')
        # ap metric for validation
        if self.pipeline.get('ap_metric'):
            metric_iou_thresh = self.pipeline['ap_metric'].get('iou_thresh', 0.5)
            self.ap_metrics = ApMetric(iou_thresh=metric_iou_thresh, class_names=self.class_names)
        self.img_names = parse_scene_file(self.data_name, cfg['scene_file'], cfg['data_dir'])
        self.logger = Logger(log_file=cfg['logger_file'])
        self.device = get_device(cfg['device'], self.logger, device_name)[0]
        self.model = self._build_model(cfg['network'], weight_file=cfg['weight_file'])
         
    def _build_model(self, model_cfg, weight_file):
        r"""Build network from model configs. """
        assert os.path.exists(weight_file)
        model = Detector(**model_cfg)
        resume_weights(model, weight_file, self.logger)
        model.to(device=self.device)
        log_printer(self.logger, 'Builded {} network model'.format(model_cfg['name']))
        model = model.eval()
        return model
    
    def _read_image_and_label(self, scene):
        image = read_image(self.data_name, scene, self.data_dir)
        label = read_label(self.data_name, scene, self.label_dir, **self.data_kwargs)
        image, label = scale_image_and_label(image, self.input_size, label=label, keep_ratio=self.keep_ratio)
        image = (image / 255.0 - self.image_mean) / self.image_std
        image = np.rollaxis(image, 2)
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)
        if self.include_bg == False and label is not None:
            label[..., -1] -= 1
        return {
            'image': image.to(self.device),
            'label': label }
        
    def _visualize_dets(self, scene, pred):
        scale = 3
        save_file = os.path.join(self.save_dir, 'visualize', scene + '.png')
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        image = read_image(self.data_name, scene, self.data_dir)
        image = scale_image_and_label(image, input_size=self.input_size, keep_ratio=self.keep_ratio)[0]
        image = cv2.resize(image, (scale*self.input_size, scale*self.input_size))
        for x1,y1,x2,y2, cls_id, score in pred:
            cls_id = int(cls_id)
            cls_name = self.class_names[cls_id]
            x1 = scale*int(min(max(x1, 0), self.input_size-1))
            y1 = scale*int(min(max(y1, 0), self.input_size-1))
            x2 = scale*int(min(max(x2, 0), self.input_size-1))
            y2 = scale*int(min(max(y2, 0), self.input_size-1))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
            text = "{}: {:.2f}".format(cls_name, score)
            cv2.putText(image, text, (int(0.8*x1+0.2*x2), int(0.5*y1+0.5*y2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imwrite(save_file, image)

    def process(self):
        def _run_model(inputs):
            with torch.no_grad():
                t_s = timer()
                y_pred = self.model(inputs, is_training=False)['res'][0]
                model_time = timer() - t_s
            return y_pred, model_time

        run_model_time = 0
        t_s = timer()
        self.img_names =  self.img_names[0:100]
        for idx, scene in enumerate(self.img_names):
            log_printer(self.logger, 'processing [{}/{}] scene_id: {}, elapsed time: {:.2f} seconds'.format(idx+1, len(self.img_names), scene, timer() - t_s))
            inputs = self._read_image_and_label(scene)
            y_pred, model_time = _run_model(inputs)
            if self.pipeline.get('ap_metric') and inputs['label'] is not None:
                self.ap_metrics.update(y_pred, inputs['label'])
            self._visualize_dets(scene, y_pred)
            run_model_time += model_time
        if self.pipeline.get('ap_metric'):
            self.ap_metrics.compute_ap_metrics(logger=self.logger)
        log_printer(self.logger, 'Average model_time is {:.3f}ms\n'.format(1000*run_model_time/len(self.img_names)))


def get_cfg_from_args(args):
    cfg_file = args.cfg
    with open(cfg_file,'rt') as f:
        cfg = yaml.full_load(f)
    if args.save_dir is not None:
        cfg['save_dir'] = args.save_dir
    if args.data_dir is not None:
        cfg['data_dir'] = args.data_dir
    if args.weight_file is not None:
        cfg['weight_file'] = args.weight_file
    if args.scene_file is not None:
        cfg['scene_file'] = args.scene_file
    logger_file = os.path.join(cfg['save_dir'], 'log.txt')
    cfg['logger_file'] = logger_file
    os.makedirs(cfg['save_dir'], exist_ok=True)
    shutil.copyfile(cfg_file, os.path.join(cfg['save_dir'], 'config_infer.yaml'))
    return cfg


def main(args):
    cfg = get_cfg_from_args(args)
    Inference = ModelInference(cfg, args.device_name)
    Inference.process()
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation Inference Pipeline')
    parser = argparse.ArgumentParser(description='PyTorch Line Detection Inference Pipeline')
    parser.add_argument("--cfg", help="path to config file", type=str, default='configure/infer/voc/voc_deeplabv3.yaml')
    parser.add_argument("--weight-file", help="resume weight file", type=str)
    parser.add_argument("--scene-file", help="image scenes list", type=str)
    parser.add_argument("--data-dir", help="input data dir", type=str)
    parser.add_argument("--save-dir", help="output save dir", type=str)
    parser.add_argument("--device-name", help="device to run", type=str, default="cuda")
    
    opts = parser.parse_args()
    opts.cfg = 'configure/infer/voc/yolo.yaml'
    #opts.cfg = 'configure/infer/voc/fpn_rcnn.yaml'
    main(opts)
