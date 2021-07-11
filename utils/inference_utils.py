# encoding: utf-8
"""
@author: Chong Li
@time: 2021/02/02 14:37
@desc:
"""

import os
from PIL import Image
import numpy as np
import cv2
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

__all__ = ["parse_scene_file", "return_class_names", "read_image", "read_label"]


def parse_voc_scene_file(scene_file, data_dir, label_dir=None):
    img_names = []
    with open(scene_file) as fp:
        scene_ids = [line.strip() for line in fp.readlines()]
    for scene in scene_ids:
        if not os.path.exists(os.path.join(data_dir, scene+'.jpg')):
            continue
        if (label_dir is not None) and (not os.path.exists(os.path.join(label_dir, scene+'.xml'))):
            continue
        img_names.append(scene)
    return img_names


def parse_scene_file(dtype, scene_file, data_dir, label_dir=None):
    if dtype == 'voc':
        return parse_voc_scene_file(scene_file, data_dir, label_dir)
    else:
        raise NotImplementedError


def return_class_names(dtype):
    if dtype == 'voc':
        return ('background', 'aeroplane', 'bicycle',
         'bird', 'boat', 'bottle', 'bus', 'car',
         'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant',
         'sheep', 'sofa', 'train', 'tvmonitor')
    else:
        raise NotImplementedError


def resize_image(image, target_size, lib='pil', interpolation_type='bilinear'):
    # image: numpy uint8 image, [0, 255], shape is [h, w, 3] or [h, w]
    if tuple(image.shape[:2]) == tuple(target_size):
        return image
    if lib == 'cv2':
        target_size = (target_size[1], target_size[0])
        if interpolation_type == 'bilinear':
            interpolation = cv2.INTER_LINEAR
        elif interpolation_type == 'bicubic':
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_NEAREST
        image = cv2.resize(image, target_size, interpolation=interpolation)
    elif lib == 'pil':
        target_size = (target_size[1], target_size[0])
        if interpolation_type == 'bilinear':  # same as cv2's bilinear resize
            interpolation = Image.BILINEAR
        elif interpolation_type == 'bicubic':
            interpolation = Image.BICUBIC
        elif interpolation_type == 'antialias':
            interpolation = Image.ANTIALIAS
        else:
            interpolation = Image.NEAREST
        # convert to Image
        image = Image.fromarray(image)  # numpy -> Image
        image = image.resize(target_size, resample=interpolation)
        image = np.array(image)  # Image -> numpy
    else:
        raise NotImplementedError
    return image


def read_voc_image(scene, data_dir):
    image_file = os.path.join(data_dir, scene + '.jpg')
    image = Image.open(image_file).convert('RGB')
    image = np.array(image)
    return image


def read_image(dtype, scene, data_dir):
    if dtype == 'voc':
        return read_voc_image(scene, data_dir)
    else:
        raise NotImplementedError


def read_voc_label(label_file, use_difficult=False):
    r"""Parse xml file and return labels."""
    def _validate_label(xmin, ymin, xmax, ymax, width, height):
        return xmin  >= 0 and ymin >= 0 and xmin < xmax < width and ymin < ymax < height
    class_names = return_class_names('voc')
    index_map = dict(zip(class_names, range(len(class_names))))
    root = ET.parse(label_file).getroot()
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    label = []
    for obj in root.iter('object'):
        try:
            difficult = int(obj.find('difficult').text)
        except ValueError:
            difficult = 0
        if not use_difficult and difficult > 0:
            continue
        cls_name = obj.find('name').text.strip().lower()
        if cls_name not in class_names:
            continue
        cls_id = index_map[cls_name]  
        xml_box = obj.find('bndbox')
        xmin = (float(xml_box.find('xmin').text) - 1)
        ymin = (float(xml_box.find('ymin').text) - 1)
        xmax = (float(xml_box.find('xmax').text) - 1)
        ymax = (float(xml_box.find('ymax').text) - 1)
        if not _validate_label(xmin, ymin, xmax, ymax, width, height):
            continue   
        label.append([xmin, ymin, xmax, ymax, cls_id])
    return np.array(label).reshape(-1, 5)
    
    
def read_label(dtype, scene, label_dir, **kwargs):
    if label_dir is None:
        return None
    if dtype == 'voc':
        label_file = os.path.join(label_dir, scene+'.xml')
        return read_voc_label(label_file, **kwargs)
    else:
        raise NotImplementedError
    

def scale_image_and_label(image, input_size, keep_ratio=False, **kwargs):
    r"""keep_ratio is True, first scale the longest side """
    image_h, image_w = image.shape[:2]
    dim_diff = np.abs(image_h - image_w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    if keep_ratio and image_h >= image_w:
        image = np.pad(image, ((0,0),(pad1,pad2), (0,0)))
    elif keep_ratio and image_w > image_h:
        image = np.pad(image, ((pad1,pad2),(0,0), (0,0)))
    else:
        pad1 = pad2 = 0
    image = resize_image(image, (input_size, input_size))
    if kwargs.get('label') is not None:
        bboxes = kwargs.get('label')
        if keep_ratio and image_h >= image_w:
            bboxes[:, 0] += pad1
            bboxes[:, 2] += pad1
            scale_w = scale_h = input_size / image_h
        elif keep_ratio and image_w > image_h:
            bboxes[:, 1] += pad1
            bboxes[:, 3] += pad1
            scale_w = scale_h = input_size / image_w
        else:
            scale_w = input_size / image_w
            scale_h = input_size / image_h
        bboxes[:, 0] = np.clip(bboxes[:, 0] * scale_w, 0, input_size - 1)
        bboxes[:, 2] = np.clip(bboxes[:, 2] * scale_w, 0, input_size - 1)
        bboxes[:, 1] = np.clip(bboxes[:, 1] * scale_h, 0, input_size - 1)
        bboxes[:, 3] = np.clip(bboxes[:, 3] * scale_h, 0, input_size - 1)
        return image, bboxes.astype(np.float32)
    else:
        return image, None
