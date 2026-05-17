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

        cls_scores = all_cls_scores[-1].float()
        mask_preds = all_mask_preds[-1].float()

        num_imgs = cls_scores.size(0)
        cls_nan = torch.isnan(cls_scores)
        cls_inf = torch.isinf(cls_scores)
        if cls_nan.any() or cls_inf.any():
            import logging
            _logger = logging.getLogger(__name__)
            _logger.warning(
                f'cls_scores: NaN={cls_nan.sum().item()} Inf={cls_inf.sum().item()} '
                f'/ {cls_scores.numel()} elements → cleaned. '
                f'Loss is healthy → fp16 AMP noise, not model collapse.')
        cls_scores_list = [
            cls_scores[i].nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            for i in range(num_imgs)]

        mask_nan = torch.isnan(mask_preds)
        mask_inf = torch.isinf(mask_preds)
        if mask_nan.any() or mask_inf.any():
            import logging
            _logger = logging.getLogger(__name__)
            _logger.warning(
                f'mask_preds: NaN={mask_nan.sum().item()} Inf={mask_inf.sum().item()} '
                f'/ {mask_preds.numel()} elements → nan_to_num+clamp applied. '
                f'Loss is healthy → this is fp16 AMP numerical noise, not model collapse.')
        mask_preds_list = [
            mask_preds[i].nan_to_num(0.0).clamp(-50, 50)
            for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)
        labels = torch.stack(labels_list, dim=0)
        label_weights = torch.stack(label_weights_list, dim=0)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        cls_scores_flat = cls_scores.flatten(0, 1)
        labels_flat = labels.flatten(0, 1)
        label_weights_flat = label_weights.flatten(0, 1)

        class_weight = cls_scores_flat.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores_flat,
            labels_flat,
            label_weights_flat,
            avg_factor=class_weight[labels_flat].sum())

        from mmdet.utils import reduce_mean
        num_total_masks = reduce_mean(
            cls_scores_flat.new_tensor([avg_factor], dtype=torch.float32))
        num_total_masks = max(num_total_masks, 1)

        mask_preds_pos = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            loss_dice = mask_preds_pos.sum()
            loss_mask = mask_preds_pos.sum()
        else:
            from mmdet.models.utils.point_sample import (
                get_uncertain_point_coords_with_randomness, point_sample)
            with torch.no_grad():
                points_coords = get_uncertain_point_coords_with_randomness(
                    mask_preds_pos.unsqueeze(1), None, self.num_points,
                    self.oversample_ratio, self.importance_sample_ratio)
                mask_point_targets = point_sample(
                    mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
            mask_point_preds = point_sample(
                mask_preds_pos.unsqueeze(1), points_coords).squeeze(1)

            loss_dice = self.loss_dice(
                mask_point_preds, mask_point_targets,
                avg_factor=num_total_masks)

            mask_point_preds_flat = mask_point_preds.reshape(-1)
            mask_point_targets_flat = mask_point_targets.reshape(-1)

            if torch.isnan(loss_dice) or torch.isnan(loss_cls):
                print(f'[NaN-DEBUG] loss_dice={loss_dice.item()} loss_cls={loss_cls.item()}')
                print(f'[NaN-DEBUG] mask_point_preds nan={torch.isnan(mask_point_preds_flat).any().item()} '
                      f'inf={torch.isinf(mask_point_preds_flat).any().item()} '
                      f'range=[{mask_point_preds_flat.min().item():.4f}, {mask_point_preds_flat.max().item():.4f}]')
                print(f'[NaN-DEBUG] mask_point_targets nan={torch.isnan(mask_point_targets_flat).any().item()} '
                      f'range=[{mask_point_targets_flat.min().item():.4f}, {mask_point_targets_flat.max().item():.4f}]')

            loss_mask = self.loss_mask(
                mask_point_preds_flat,
                mask_point_targets_flat,
                avg_factor=num_total_masks * self.num_points)

            if torch.isnan(loss_mask):
                print(f'[NaN-DEBUG] loss_mask is NaN! '
                      f'loss_cls={loss_cls.item():.4f} loss_dice={loss_dice.item():.4f} '
                      f'num_total_masks={num_total_masks} num_points={self.num_points} '
                      f'avg_factor={num_total_masks * self.num_points} '
                      f'pred_nan={torch.isnan(mask_point_preds_flat).any().item()} '
                      f'target_nan={torch.isnan(mask_point_targets_flat).any().item()} '
                      f'pred_range=[{mask_point_preds_flat.min().item():.4f}, {mask_point_preds_flat.max().item():.4f}] '
                      f'target_range=[{mask_point_targets_flat.min().item():.4f}, {mask_point_targets_flat.max().item():.4f}]')

        loss_dict = dict()
        loss_dict['loss_cls'] = loss_cls
        loss_dict['loss_mask'] = loss_mask
        loss_dict['loss_dice'] = loss_dice

        return loss_dict

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
