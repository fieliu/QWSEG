import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS


def get_one_hot(label, N_cls):
    return F.one_hot(label, num_classes=N_cls).permute(0, 1, 2, 3).float()


@MODELS.register_module()
class RegionL1(nn.Module):
    def __init__(self, tau=1.0, loss_weight=1.0, N_cls=None, loss_name='rl1_loss'):
        super().__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.N_cls = N_cls
        self._loss_name = loss_name

    def forward(self, preds_S, preds_T):
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, H, W = preds_S.shape
        one_hot = get_one_hot(preds_S.argmax(1), self.N_cls)
        one_hot_encoding = one_hot.contiguous().permute(0, 3, 1, 2)
        region_S = preds_S * one_hot_encoding
        region_T = preds_T * one_hot_encoding
        loss = F.l1_loss(region_S, region_T.clone().detach())
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name
