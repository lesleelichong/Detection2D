# encoding: utf-8
"""
@author: Chong Li
@time: 2021/05/14 10:21 morning
@desc:
"""
import math
import numpy as np
import torch
from utils.bbox_utils import BBoxCornerToCenter


def generate_one_stage_anchors(stride, anchor_sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
    r"""Generates a matrix of anchor boxes in [N ,(xmin, ymin, xmax, ymax)] format. 
    Anchors are centered on (stride - 1) / 2. """
    ratios = np.array(aspect_ratios, dtype=np.float)
    center = (stride - 1) / 2.0
    anchors = []
    for size in anchor_sizes:
        size_ratios = size * size / ratios
        ws = np.sqrt(size_ratios)
        hs = ws * ratios
        ws = ws[:, np.newaxis]
        hs = hs[:, np.newaxis]
        anchors.append(
            np.concatenate([
            center - 0.5 * (ws - 1),
            center - 0.5 * (hs - 1),
            center + 0.5 * (ws - 1),
            center + 0.5 * (hs - 1)], axis=-1))
    return np.concatenate(anchors, axis=0)
    

def generate_anchors(strides=[16], anchor_sizes=([128, 256, 512]), 
                     aspect_ratios=(0.5, 1, 2), alloc_size=(128,128)):
    r"""Pre generate all FPN stages anchors, image max_size is max(strides)*alloc_size
    The anchors is (center_x, center_y, width, height) style.
    """
    assert len(strides) == len(anchor_sizes)
    stage_anchors = []
    for stride, sizes in zip(strides, anchor_sizes):
        anchors = generate_one_stage_anchors(stride, sizes, aspect_ratios)
        height, width = alloc_size
        offset_x = np.arange(0, width * stride, stride)
        offset_y = np.arange(0, height * stride, stride)
        offset_x, offset_y = np.meshgrid(offset_x, offset_y)
        offsets = np.stack((offset_x.flatten(), offset_y.flatten(),
                            offset_x.flatten(), offset_y.flatten()), axis=1)
        anchors = (anchors.reshape((1, -1, 4)) + offsets.reshape((-1, 1, 4)))
        anchors = anchors.reshape((height, width, -1, 4)).astype(np.float32)
        stage_anchors.append(anchors)
    return stage_anchors


def get_batch_scores_top_n_idx(objectness, num_anchors_per_level, max_top_n):
    top_n_idx_per_stage = []
    offset = 0
    for ob in objectness.split(num_anchors_per_level, 1):    
        num_anchors = ob.shape[1]
        top_n = min(max_top_n, num_anchors)
        top_n_idx = ob.topk(top_n, dim=1)[1]
        top_n_idx_per_stage.append(top_n_idx + offset)
        offset += num_anchors
    return torch.cat(top_n_idx_per_stage, dim=1)
   
 
class NormalizedProposalDecoder(object):
    r"""Decode bounding boxes.
    This decoder must cooperate with NormalizedProposalEncoder of same `stds/means`
    in order to get properly reconstructed bounding boxes.
    Returned bounding boxes are using corner type: `x_{min}, y_{min}, x_{max}, y_{max}`.
    Parameters
    ----------
    stds : array-like of size 4
        Std value to be divided from encoded values, default is (0.1, 0.1, 0.2, 0.2).
    means : array-like of size 4
        Mean value to be subtracted from encoded values, default is (0., 0., 0., 0.).
    bbox_size_clip: Clip bounding box prediction to to prevent exponentiation from overflowing.
    """

    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.), bbox_size_clip=6):
        super(NormalizedProposalDecoder, self).__init__()
        self.stds = stds
        self.means = means
        self.bbox_size_clip = bbox_size_clip
        
    def __call__(self, bboxes, anchors):
        anchors = BBoxCornerToCenter(anchors)
        # Denormalization for (dx, dy, dw, dh)
        for idx in range(4):
            bboxes[..., idx] = bboxes[..., idx] * self.stds[idx] + self.means[idx]
        ctr_x = bboxes[..., 0] * anchors[..., 2] + anchors[..., 0]
        ctr_y = bboxes[..., 1] * anchors[..., 3] + anchors[..., 1]
        dw = torch.clamp(bboxes[..., 2], max=self.bbox_size_clip)
        dh = torch.clamp(bboxes[..., 3], max=self.bbox_size_clip)
        dw = torch.exp(dw) * anchors[..., 2]
        dh = torch.exp(dh) * anchors[..., 3]
        ctr_x = ctr_x.unsqueeze(-1)
        ctr_y = ctr_y.unsqueeze(-1)
        dw = dw.unsqueeze(-1)
        dh = dh.unsqueeze(-1)  
        proposals = torch.cat([ctr_x - 0.5 * (dw - 1),
                   ctr_y - 0.5 * (dh - 1),
                   ctr_x + 0.5 * (dw - 1),
                   ctr_y + 0.5 * (dh - 1)], dim=-1)
        return proposals


class NormalizedProposalEncoder(object):
    r"""Eecode bounding boxes.
    Returned bounding boxes are using center type. {x_ctr, y_ctr, width, height}
    """
    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.)):
        super(NormalizedProposalEncoder, self).__init__()
        self.stds = stds
        self.means = means
        
    def __call__(self, bboxes, anchors):
        anchors = BBoxCornerToCenter(anchors)
        bboxes = BBoxCornerToCenter(bboxes)
        ex_ctr_x = anchors[..., 0]
        ex_ctr_y = anchors[..., 1]
        ex_widths = anchors[..., 2]
        ex_heights = anchors[..., 3]
        gt_ctr_x = bboxes[..., 0]
        gt_ctr_y = bboxes[..., 1]
        gt_widths = bboxes[..., 2]
        gt_heights = bboxes[..., 3]
        dx = (gt_ctr_x - ex_ctr_x) / ex_widths.clamp(min=1e-6)
        dy = (gt_ctr_y - ex_ctr_y) / ex_heights.clamp(min=1e-6)
        dw = torch.log(gt_widths.clamp(min=1e-6) / ex_widths.clamp(min=1e-6))
        dh = torch.log(gt_heights.clamp(min=1e-6) / ex_heights.clamp(min=1e-6))
        proposals = torch.cat([dx.unsqueeze(-1),
                               dy.unsqueeze(-1),
                               dw.unsqueeze(-1),
                               dh.unsqueeze(-1)], dim=-1)
        # Normalization for (dx, dy, dw, dh)
        for idx in range(4):
            proposals[..., idx] = (proposals[..., idx] - self.means[idx]) / self.stds[idx]
        return proposals


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.
    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.
    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                _set_low_quality_matches for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.
        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None
        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold )
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS
        if self.allow_low_quality_matches:
            assert all_matches is not None
            self._set_low_quality_matches(matches, all_matches, match_quality_matrix)
        return matches

    def _set_low_quality_matches(self, matches, all_matches, match_quality_matrix):
        r"""
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt = match_quality_matrix.max(dim=1)[0]
        # Find highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.where(
            match_quality_matrix == highest_quality_foreach_gt[:, None])
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]

class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """
    def __init__(self, num_sample_per_image, positive_fraction):
        """
        Arguments:
            num_sample_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.num_sample_per_image = num_sample_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.
        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])
        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.where(matched_idxs_per_image >= 1)[0]
            negative = torch.where(matched_idxs_per_image == 0)[0]
            num_pos = int(self.num_sample_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.num_sample_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)
            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]
            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1
            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)
        return pos_idx, neg_idx
