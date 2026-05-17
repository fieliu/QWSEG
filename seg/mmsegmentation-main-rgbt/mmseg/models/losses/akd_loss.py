import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS


class Feature_Pool(nn.Module):
    def __init__(self, dim, ratio=16):
        super().__init__()
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.gmp_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(1, dim // ratio)
        self.down = nn.Linear(dim, hidden)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(hidden, dim)

    def forward(self, x):
        pooled = (self.gap_pool(x) + self.gmp_pool(x)) * 0.5
        v = pooled.flatten(1)
        y = self.up(self.act(self.down(v)))
        y = y / (y.norm(dim=1, keepdim=True) + 1e-6)
        return y


@MODELS.register_module()
class AKDLoss(nn.Module):
    def __init__(self,
                 tau=1.0,
                 loss_weight=1.0,
                 ratio=16,
                 loss_name='loss_akd'):
        super().__init__()
        self.tau = float(tau)
        self.loss_weight = float(loss_weight)
        self.ratio = int(ratio)
        self._loss_name = loss_name
        self.pools = None
        self._lazy_built = False

    def _lazy_build(self, preds_S):
        dims = [p.shape[1] for p in preds_S]
        dev = preds_S[0].device
        self.pools = nn.ModuleList([Feature_Pool(c, self.ratio).to(dev) for c in dims])
        self._lazy_built = True

    def forward(self, preds_T, preds_S):
        assert isinstance(preds_T, (list, tuple)) and isinstance(preds_S, (list, tuple))
        assert len(preds_T) == len(preds_S)

        if not self._lazy_built or self.pools is None:
            self._lazy_build(preds_S)

        losses = []
        for t, s, pool in zip(preds_T, preds_S, self.pools):
            assert s.dim() == 4 and t.dim() == 4
            assert s.shape[1] == t.shape[1]
            w_s = pool(s)
            w_t = pool(t)
            alpha = torch.sigmoid(s.shape[1] * (w_s * w_t))[:, :, None, None]
            loss_i = F.mse_loss(s * alpha, t * alpha)
            losses.append(loss_i)

        loss = torch.stack(losses).mean()
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name
