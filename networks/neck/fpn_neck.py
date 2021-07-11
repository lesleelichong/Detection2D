# encoding: utf-8
"""
@author: Chong Li
@time: 2021/06/29 17:26
@desc:
"""

from torch import nn
from networks.conv_utils import Conv


class FeaturePyramidNetwork(nn.Module):
    r"""
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
    """
    def __init__(self, in_channels, out_channels, upsample_mode='nearest'):
        super(FeaturePyramidNetwork, self).__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels]
        if upsample_mode == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_ch in in_channels:
            inner_block_module = Conv(in_ch, out_channels, kernel_size=1)
            layer_block_module = Conv(out_channels, out_channels, kernel_size=3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

    def forward(self, layers):
        r"""
        Computes the FPN for a set of feature maps.
        Arguments:
            layers: feature maps for each feature level.
            They are ordered from smallest resolution first.      
        """
        results = []
        last_inner = self.inner_blocks[0](layers[0])
        results.append(self.layer_blocks[0](last_inner))
        for idx in range(1, len(self.inner_blocks)):
            inner_lateral = self.inner_blocks[idx](layers[idx])
            inner_top_down = self.up(last_inner)
            last_inner = inner_lateral + inner_top_down
            results.append(self.layer_blocks[idx](last_inner))
        return results
