import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss


def modal_cross_entropy(pred,
                        pred_2,
                        weight=None,
                        class_weight=None,
                        reduction='mean',
                        avg_factor=None,
                        ignore_index=-100,
                        avg_non_ignore=False):
    pred_softmax = F.softmax(pred_2, dim=1)
    predicted_labels = torch.argmax(pred_softmax, dim=1)

    loss = F.cross_entropy(
        pred,
        predicted_labels,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = predicted_labels.numel() - (predicted_labels == ignore_index).sum().item()
    if weight is not None:
        weight = weight.float()

    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@MODELS.register_module()
class M_CrossEntropyLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_ce_m',
                 avg_non_ignore=False):
        super().__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        self.cls_criterion = modal_cross_entropy
        self._loss_name = loss_name

    def extra_repr(self):
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                cls_score2,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            cls_score2,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            avg_non_ignore=self.avg_non_ignore,
            ignore_index=ignore_index,
            **kwargs)
        return loss_cls

    @property
    def loss_name(self):
        return self._loss_name
