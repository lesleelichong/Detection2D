# encoding: utf-8
"""
@author: Chong Li
@time: 2021/05/27 18:39
@desc:
"""

from torch import nn


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1):
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(Conv, self).__init__(*modules)
