# encoding: utf-8
"""
@author: Chong Li
@time: 2021/05/27 18:39
@desc:
"""

import torch
from torch import nn
from torch.nn import functional as F
from networks.conv_utils import Conv


class UpConv(nn.Module):
    def __init__(self, in_channels, upsample_mode='bilinear'):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if upsample_mode == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce_conv = Conv(in_channels,in_channels // 2, 1)
        
    def forward(self, x1, x2):
        """
        :param x1: low resolution feature map
        :param x2: high resolution feature map
        :return:
        """
        x1 = self.reduce_conv(x1)
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return x
    
 
class NeckConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NeckConv, self).__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2 = Conv(out_channels, 2*out_channels, kernel_size=3, padding=1)
        self.conv3 = Conv(2*out_channels, out_channels, kernel_size=1, padding=0)
        self.conv4 = Conv(out_channels, 2*out_channels, kernel_size=3, padding=1)
        self.conv5 = Conv(2*out_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
    

class YoloNeck(nn.Module):
    def __init__(self, in_channels, unified_channels):
        super(YoloNeck, self).__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels]
        assert len(in_channels) <= len(unified_channels)
        if in_channels != unified_channels:
            unified_convs = [Conv(x, y, 3, padding=1) for x, y in zip(in_channels, unified_channels[:len(in_channels)])]
            self.unified_convs = nn.ModuleList(unified_convs)
        else:
            self.unified_convs = None
            
        up_convs = []
        for i in range(1, len(in_channels)):
            in_ch = unified_channels[i]
            up_convs.append(UpConv(in_ch))
        self.up_convs = nn.ModuleList(up_convs) if len(up_convs) != 0 else None
        
        unified_channels= [0] + unified_channels
        neck_convs = []
        for i in range(len(in_channels)):
            in_ch = unified_channels[i] // 4 + unified_channels[i+1]
            out_ch = unified_channels[i+1] // 2
            neck_convs.append(NeckConv(in_ch, out_ch))
        self.neck_convs = nn.ModuleList(neck_convs)
            
    def forward(self, layers):
        if self.unified_convs is None:
            unified_layers = [layer for layer in layers]
        else:
            unified_layers = [conv(layer) for layer, conv in zip(layers, self.unified_convs)]
         
        out_layers = [self.neck_convs[0](unified_layers[0])]
        if self.up_convs is not None:
            for idx, up_conv in enumerate(self.up_convs):
                x1 = out_layers[idx]
                x2 = unified_layers[idx+1]
                out_layers.append(self.neck_convs[idx+1](up_conv(x1, x2)))
        return out_layers
