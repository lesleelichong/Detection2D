# encoding: utf-8
"""
@author: Chong Li
@time: 2021/05/27 18:39
@desc:
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from networks.conv_utils import Conv


class YoloHead(nn.Module):
    def __init__(self, in_channels, class_num, anchors, strides, alloc_size=(128, 128)):
        super(YoloHead, self).__init__()
        if isinstance(strides, int):
            strides = [strides]
        self.strides = strides
        self.class_num = class_num
        #every stage should have the same num_anchors
        self.num_anchors = len(anchors) // len(in_channels)
        self.anchors = [anchors[i*self.num_anchors:(i+1)*self.num_anchors] for i in range(len(anchors)//self.num_anchors)]
        self.num_pred = 4 + 1 + class_num
        grid_x, grid_y = np.meshgrid(np.arange(alloc_size[1]), np.arange(alloc_size[0]))  
        self.offsets = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)
        
        head_convs = [Conv(in_ch, 2*in_ch, 3, padding=1) for in_ch in in_channels]
        self.head_convs = nn.ModuleList(head_convs)
        # The end of the network, should no BN and RELU   
        pred_convs = [nn.Conv2d(2*in_ch,self.num_pred*self.num_anchors,kernel_size=1,stride=1,bias=True) for in_ch in in_channels]
        self.pred_convs = nn.ModuleList(pred_convs)
        
    def forward(self, layers, is_training=False):
        head_layers = [conv(layer) for layer, conv in zip(layers, self.head_convs)]
        pred_layers = [conv(layer) for layer, conv in zip(head_layers, self.pred_convs)]
        if is_training == True:
            return self._forward_train(pred_layers)
        else:
            return self._forward_test(pred_layers)

    def _forward_train(self, pred_layers):
        anchor_list = []
        bbox_list = []
        raw_box_centers = []
        raw_box_scales = []
        objness = []
        class_pred = []
        anchor_index_map = []
        start_index = 0
        
        for layer, anchor, stride in zip(pred_layers, self.anchors[::-1], self.strides[::-1]):
            offset = self.offsets[0:layer.shape[3], 0:layer.shape[2]]
            offset = torch.from_numpy(offset.reshape(1,-1,1,2)).to(layer.device)
            anchor = torch.from_numpy(np.array(anchor).reshape(1,1,-1,2)).to(layer.device)
            b, c, h, w = layer.shape
            # use it to search the gt match anchor's fpn layer and location.
            anchor_index_map += [{'start_index': start_index, 'width':w, 'height':h, 'stride':stride} 
                                 for _ in range(self.num_anchors)]
            start_index += h * w
            layer = layer.reshape(b, c, -1)
            pred = layer.permute(0, 2, 1).reshape(b, h * w, self.num_anchors, self.num_pred)
            anchor_list.append(anchor)
            raw_box_centers.append(pred[:,:,:,:2])
            raw_box_scales.append(pred[:,:,:,2:4])
            objness.append(pred[:,:,:,4].reshape(b, h * w, -1, 1))
            class_pred.append(pred[:,:,:,5:])
            bbox_center = (torch.sigmoid(pred[:,:,:,:2]) + offset) * stride
            bbox_scale = torch.exp(pred[:,:,:,2:4]) * anchor
            half_wh = bbox_scale / 2.0
            bbox = torch.cat([bbox_center-half_wh, bbox_center+half_wh], dim=-1)
            bbox_list.append(bbox)
            
        return { 'bboxes_pred' : torch.cat(bbox_list, dim=1),
                 'raw_box_centers': torch.cat(raw_box_centers, dim=1),
                 'raw_box_scales': torch.cat(raw_box_scales, dim=1),
                 'objness_pred' : torch.cat(objness, dim=1),
                 'class_pred' : torch.cat(class_pred, dim=1),
                 'anchors' : torch.cat(anchor_list, dim=-2), 
                 'anchor_index_map' : anchor_index_map,
                 'stage_num_anchors' : self.num_anchors }

    def _forward_test(self, pred_layers):
        r"""
        Return: An Tensor (B, N, [xmin，ymin,xmax,ymax, class, score])
        """
        bbox_list = []
        score_list = []
        for layer, anchor, stride in zip(pred_layers, self.anchors[::-1], self.strides[::-1]):
            offset = self.offsets[0:layer.shape[3], 0:layer.shape[2]]
            offset = torch.from_numpy(offset.reshape(1,-1,1,2)).to(layer.device)
            anchor = torch.from_numpy(np.array(anchor).reshape(1,1,-1,2)).to(layer.device)
            b, c, h, w = layer.shape
            layer = layer.reshape(b, c, -1)
            pred = layer.permute(0,2,1).reshape(b, h*w, self.num_anchors, self.num_pred)
            confidence = torch.sigmoid(pred[:,:,:,4].reshape(b,h*w,-1,1))
            class_pred = torch.sigmoid(pred[:,:,:,5:])
            class_score = class_pred * confidence
            score_list.append(class_score.reshape(b,-1,self.class_num))
            bbox_center = (torch.sigmoid(pred[:,:,:,:2]) + offset) * stride
            bbox_scale = torch.exp(pred[:,:,:,2:4]) * anchor
            half_wh = bbox_scale / 2.0
            bbox = torch.cat([bbox_center-half_wh, bbox_center+half_wh], dim=-1)
            bbox_list.append(bbox.reshape(b,-1,4))
    
        bboxes_score, bboxes_class = torch.max(torch.cat(score_list, dim=1), dim=-1, keepdim=True)
        bboxes_pred = torch.cat(bbox_list, dim=1)
        # (B, N, [xmin，ymin,xmax,ymax,class, score])
        res = torch.cat([bboxes_pred, bboxes_class, bboxes_score], dim=-1)  
        return res
     