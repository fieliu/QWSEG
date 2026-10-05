from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

try:
    from mmdet.models.dense_heads import \
        Mask2FormerHead as MMDET_Mask2FormerHead
except ModuleNotFoundError:
    MMDET_Mask2FormerHead = BaseModule

from mmengine.structures import InstanceData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils import ConfigType, SampleList


@MODELS.register_module()
class Mask2FormerHead(MMDET_Mask2FormerHead):

    def __init__(self,
                 num_classes,
                 align_corners=False,
                 ignore_index=255,
                 **kwargs):
        kwargs['num_things_classes'] = 0
        kwargs['num_stuff_classes'] = num_classes
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes
        self.ignore_index = ignore_index

    def _get_targets_single(self, cls_score, mask_pred, gt_instances, img_meta):
        # Hungarian matching is non-differentiable and must stay in fp32.  A
        # plain ``.float()`` is not enough here: the surrounding AMP context
        # would autocast the 12,544-point cost einsums back to fp16, whose
        # reductions can overflow and make SciPy report an infeasible matrix.
        mask_pred = mask_pred.float().nan_to_num(0.0).clamp(-50, 50)
        cls_score = cls_score.float().nan_to_num(
            nan=0.0, posinf=0.0, neginf=0.0)
        with torch.autocast(device_type=cls_score.device.type, enabled=False):
            return super()._get_targets_single(
                cls_score, mask_pred, gt_instances, img_meta)

    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        batch_img_metas = []
        batch_gt_instances = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt_sem_seg = data_sample.gt_sem_seg.data
            classes = torch.unique(
                gt_sem_seg,
                sorted=False,
                return_inverse=False,
                return_counts=False)

            gt_labels = classes[classes != self.ignore_index]

            masks = []
            for class_id in gt_labels:
                masks.append(gt_sem_seg == class_id)

            if len(masks) == 0:
                gt_masks = torch.zeros(
                    (0, gt_sem_seg.shape[-2],
                     gt_sem_seg.shape[-1])).to(gt_sem_seg).long()
            else:
                gt_masks = torch.stack(masks).squeeze(1).long()

            instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
            batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        all_cls_scores = [score.float() for score in all_cls_scores]
        all_mask_preds = [pred.float() for pred in all_mask_preds]
        if not all(torch.isfinite(score).all() for score in all_cls_scores):
            raise FloatingPointError('non-finite Mask2Former class logits')
        if not all(torch.isfinite(pred).all() for pred in all_mask_preds):
            raise FloatingPointError('non-finite Mask2Former mask logits')

        # Keep matching and point losses in fp32, and retain Mask2Former's
        # auxiliary supervision for every transformer-decoder layer.  The old
        # implementation trained only the final layer, which materially slowed
        # early convergence and was not equivalent to standard Mask2Former.
        with torch.autocast(
                device_type=all_cls_scores[0].device.type, enabled=False):
            return self.loss_by_feat(
                all_cls_scores, all_mask_preds,
                batch_gt_instances, batch_img_metas)

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1].float()
        mask_pred_results = all_mask_preds[-1].float()
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits
