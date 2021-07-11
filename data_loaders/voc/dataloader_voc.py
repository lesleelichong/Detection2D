
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset as TorchDataset
from data_loaders.augmentation_2d import Augmentation2D
from utils.inference_utils import parse_scene_file, read_voc_label, scale_image_and_label
from utils.log import log_printer


class Dataset(TorchDataset):
    def __init__(self, config, is_train=False, logger=None):
        super(Dataset, self).__init__()
        self.data_name = 'voc'
        self.logger = logger
        self.is_train = is_train
        self.iter = 0
        self.keep_ratio = config.get('keep_ratio', False)
        self.max_bbox_num = config.get('max_bbox_num', 20)
        self.image_mean = config.get('image_mean', 0.5)
        self.image_std = config.get('image_std', 1.0)
        self.use_difficult = config.get('use_difficult', False)
        self.preload_label = config.get('preload_label', False)
        self.multiscale_interval = config.get('multiscale_interval', 1)
        self.augmentation = None
        if is_train and config.get('augmentation') is not None:
            self.augmentation = Augmentation2D(**config['augmentation'])
        prefix = 'train' if is_train else 'val'
        input_size = config['{}_input_size'.format(prefix)]
        if isinstance(input_size, int):
            input_size = [input_size]
        self.input_size_list = input_size
        self.input_size = input_size[0]
        data_dir_list = config['{}_data_dir'.format(prefix)]
        label_dir_list = config['{}_label_dir'.format(prefix)]
        scene_file_list = config['{}_lst'.format(prefix)]
        self.image_files, self.label_files = self._parse_input_files(data_dir_list, label_dir_list, scene_file_list)
        self._label_cache = self._preload_labels() if self.preload_label else None
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = self._read_image(idx)
        label = self._label_cache[idx] if self.preload_label else self._read_label(idx)
        if self.augmentation is not None:
            image, label = self.augmentation(image, label)
        return image, label, self.image_files[idx]
        
    def _parse_input_files(self, data_dir_list, label_dir_list, scene_file_list):
        assert len(data_dir_list) == len(label_dir_list) == len(scene_file_list)
        image_files = []
        label_files = []
        for data_dir, label_dir, scene_file in zip(data_dir_list, label_dir_list, scene_file_list):
            img_names = parse_scene_file(self.data_name, scene_file, data_dir, label_dir)
            for name in img_names:
                image_files.append(os.path.join(data_dir, name+'.jpg'))
                label_files.append(os.path.join(label_dir, name+'.xml'))
        return image_files, label_files 

    def _read_image(self, idx):
        image_file = self.image_files[idx]
        image = Image.open(image_file).convert('RGB')
        image = np.array(image).astype(np.float32)
        return image

    def _read_label(self, idx):
        label_file = self.label_files[idx]
        label = read_voc_label(label_file, use_difficult=self.use_difficult)
        return label
    
    def _preload_labels(self):
        r"""Preload all labels into memory."""
        log_printer(self.logger, 'Preloading {} labels into memory...'.format(len(self)))
        return [self._read_label(idx) for idx in range(len(self))]

    def _image_transform(self, image):
        image = image / 255.
        image = (image - self.image_mean) / self.image_std
        image = np.rollaxis(image, 2)
        image = torch.from_numpy(image).float()
        return image
    
    def collate_fn(self, batch):
        batch_images = []
        batch_bboxes = []
        bboxes_num = []
        scene_list = []
        self.iter = (self.iter + 1) % ( 1 + int(len(self) / len(batch)))
        if self.iter % self.multiscale_interval == 0:
            rand_n = np.random.randint(0, len(self.input_size_list))
            self.input_size = self.input_size_list[rand_n]
            log_printer(self.logger, 'Iter {} multi scale input_size changed to {} pixels'.format(self.iter, self.input_size))
        max_bbox_num = self.max_bbox_num
        for image, bboxes, scene_file in zip(*list(zip(*batch))):
            image = image.astype(np.uint8)
            image, bboxes = scale_image_and_label(image, self.input_size, label=bboxes, keep_ratio=self.keep_ratio)
            image = self._image_transform(image)
            np.random.shuffle(bboxes)
            if bboxes.shape[0] > max_bbox_num:
                bboxes = bboxes[:max_bbox_num]
            bboxes_num.append(bboxes.shape[0])
            unified_bboxes = np.zeros([max_bbox_num, 5], np.float)
            unified_bboxes[:bboxes.shape[0]] = bboxes
            batch_bboxes.append(unified_bboxes)
            batch_images.append(image)
            scene_list.append(scene_file)
        image = torch.stack(batch_images)
        batch_bboxes = np.stack(batch_bboxes, axis=0)
        bboxes = torch.from_numpy(batch_bboxes).float()
        return image, bboxes, bboxes_num, scene_list
