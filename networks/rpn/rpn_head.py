# encoding: utf-8
"""
@author: Chong Li
@time: 2021/06/27 18:39
@desc:
"""

from torch import nn


class StandardRPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors, mode_type='sigmoid'):
        super(StandardRPNHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 
                      kernel_size=3, stride=1, 
                      padding=1, bias=True),
            nn.ReLU())
        #self.num_anchors = num_anchors
        self.crit = nn.Sigmoid() if mode_type != 'softmax' else nn.Softmax()
        # use sigmoid instead of softmax, reduce channel numbers
        # if use softmax here, then the cls_pred will num_anchors*2 output channel
        scale = 1 if mode_type != 'softmax' else 2
        self.cls_pred = nn.Conv2d(in_channels, num_anchors * scale, 
                                  kernel_size=1, stride=1, bias=True)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 
                                   kernel_size=1, stride=1, bias=True)
    
    def forward(self, layers):
        raw_rpn_scores = []
        raw_rpn_boxes = []
        rpn_scores = []
        rpn_boxes = []
        for feature in layers:
            feature = self.conv(feature)
            score = self.cls_pred(feature).permute(0,2,3,1)
            loc = self.bbox_pred(feature).permute(0,2,3,1).reshape(score.shape + (4,))
            raw_rpn_scores.append(score)
            raw_rpn_boxes.append(loc)
            rpn_scores.append(self.crit(score.detach()))
            rpn_boxes.append(loc.detach())
        # return raw predictions as well in training for bp
        return rpn_scores, rpn_boxes, raw_rpn_scores, raw_rpn_boxes
   

def rpn_head(head_type, in_channels, num_anchors, mode_type):
    if head_type == 'standard':
        return StandardRPNHead(in_channels, num_anchors, mode_type)
    else:
        return StandardRPNHead(in_channels, num_anchors, mode_type)
