# encoding: utf-8
"""
@author: Chong Li
@time: 2021/05/26 15:28
@desc:
"""

import random
import numpy as np
import cv2
from PIL import Image, ImageFilter


class Augmentation2D:
    """ 2D Detection augmentation for image and label
    """
    def __init__(self,
                 flip_h=False,
                 flip_v=False,
                 saturation=False,
                 hue=False,
                 contrast=False,
                 crop=False,
                 gauss=False,
                 probability=0.5):
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.saturation = saturation
        self.hue = hue
        self.contrast = contrast
        self.crop = crop
        self.gauss = gauss
        self.probability = probability

    def _horizontal_flip(self, image, label):
        image = image[::, ::-1, :]
        xmax = (image.shape[1] - 1) - label[:, 0]
        xmin = (image.shape[1] - 1) - label[:, 2]
        label[:, 0] = xmin
        label[:, 2] = xmax
        return image, label

    def _vertical_flip(self, image, label):
        image = image[::-1, :, :]
        ymax = (image.shape[0] - 1) - label[:, 1]
        ymin = (image.shape[0] - 1) - label[:, 3]
        label[:, 1] = ymin
        label[:, 3] = ymax
        return image, label
    
    def _saturation(self, image, label):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_image[:, :, 1] *= random.uniform(0.5, 1.5)
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        image = np.clip(image, 0, 255)
        return image, label
    
    def _contrast(self, image, label):
        image *= random.uniform(0.5, 1.5)
        image = np.clip(image, 0, 255)
        return image, label
        
    def _hue(self, image, label):
        hue_delta = 18
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_image[:, :, 0] += random.uniform(-hue_delta, hue_delta)
        hsv_image[:, :, 0][hsv_image[:, :, 0] > 360] -= 360
        hsv_image[:, :, 0][hsv_image[:, :, 0] < 0] += 360
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        image = np.clip(image, 0, 255)
        return image, label
    
    def _crop(self, image, label):
        # crop: don't crop the bbox 
        height, width = image.shape[:2]
        left_max = label[..., 0].min()
        up_max = label[..., 1].min()
        right_min = label[..., 2].max()
        below_min = label[..., 3].max()
        left = int(left_max * random.uniform(0.2, 0.8))
        up = int(up_max * random.uniform(0.2, 0.8))
        right = int(right_min + (width - right_min) * random.uniform(0.2, 0.8))
        below = int(below_min + (height - below_min) * random.uniform(0.2, 0.8))
        image = image[up:below, left:right]
        label[..., 0] = np.clip(label[..., 0] - left, 0, right - left)
        label[..., 1] = np.clip(label[..., 1] - up, 0, below - up)
        label[..., 2] = np.clip(label[..., 2] - left, 0, right - left)
        label[..., 3] = np.clip(label[..., 3] - up, 0, below - up)
        return image, label
    
    def _gauss(self, image, label):
        blur_value = np.random.uniform(0, 4)
        image = Image.fromarray(image.astype(np.uint8))
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_value))
        image = np.array(image).astype(np.float32)
        return image, label
    
    def _need_aug(self):
        return random.random() >= self.probability

    def process(self, image, label):
        if self.flip_h and self._need_aug():
            image, label = self._horizontal_flip(image, label) 
        if self.flip_v and self._need_aug():
            image, label = self._vertical_flip(image, label)
        if self.saturation and self._need_aug():
            image, label = self._saturation(image, label)
        if self.hue and self._need_aug():
            image, label = self._hue(image, label)
        if self.contrast and self._need_aug():
            image, label = self._contrast(image, label)
        if self.crop and self._need_aug():
            image, label = self._crop(image, label)
        if self.gauss and self._need_aug():
            image, label = self._gauss(image, label)
        return image, label

    def __call__(self, image, label):
        #label [N, 5(x,y,x,y,cls)]
        return self.process(image, label)


from PIL import Image
img_list = ['2007_000170.jpg',
            '2007_000175.jpg',
            '2007_000187.jpg',
            '2007_000241.jpg',
            '2007_000243.jpg',
            '2007_000250.jpg',
            '2007_000256.jpg']

'''
func = Augmentation(crop=True)
for idx, image_file in  enumerate(img_list):
    image = Image.open(image_file).convert('RGB')
    ori_image = np.array(image).astype(np.float32)
    image, label = func(ori_image, label=None)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_RGB2BGR)
    #image = np.concatenate([ori_image, image], axis=1)
    cv2.imwrite('{}.png'.format(idx), image)
bb = 9
'''

