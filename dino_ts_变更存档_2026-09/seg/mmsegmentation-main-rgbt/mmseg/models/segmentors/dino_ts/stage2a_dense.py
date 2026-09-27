"""Stage 2A: Dense Robust Warm-up (doc section 7).

Trains a full-token dense multimodal segmentation model WITHOUT any teacher, so
it gains basic semantics, fusion, and degradation handling. Loss (doc 7.4):

    L_stage2A = L_seg(f(x), y)
              + lambda_deg     * L_seg(f(C(x)), y)
              + lambda_missing * L_seg(f(Drop(x)), y)

C(x): label-preserving random degradation; Drop(x): whole-modality missing.
Real GT is reused for all three views (procedural degradation preserves scene
semantics and the output coordinate system) — this is NOT a pseudo-label.

Degradation views come from the project's DegradationGenerator (missing /
local-missing + continuous multi-level), applied INTERNALLY on the normalized
tensors (Paradigm One: single model, multiple forward passes).
"""
import torch

from mmseg.registry import MODELS
from .base_dino_ts import DinoTSBase
from ..degradation import DegradationGenerator


@MODELS.register_module()
class DinoTSDense(DinoTSBase):
    def __init__(self, *args,
                 lambda_deg: float = 1.0,
                 lambda_missing: float = 1.0,
                 degradation=None,
                 **kwargs):
        # dense token path for Stage 2A/2B
        kwargs.setdefault('forward_mode', 'dense')
        super().__init__(*args, **kwargs)
        self.lambda_deg = lambda_deg
        self.lambda_missing = lambda_missing
        self.degrader = DegradationGenerator(**(degradation or {}))
        self.current_epoch = 0

    def _seg_loss(self, inputs, data_samples, prefix, availability=None):
        feats = self.extract_feat(inputs, availability=availability)
        losses = self.decode_head.loss(feats, data_samples, self.train_cfg)
        # rename decode.* -> <prefix>.decode.* to keep the three views distinct
        return {f'{prefix}.{k}': v for k, v in losses.items()}

    @torch.no_grad()
    def _make_degraded(self, inputs):
        """C(x): degrade one modality (missing/local-missing) per sample."""
        rgb, thermal = self._split(inputs)
        drgb, dthr, _, _ = self.degrader(rgb, thermal, epoch=self.current_epoch)
        return torch.cat([drgb, dthr], dim=1)

    @torch.no_grad()
    def _make_missing(self, inputs):
        """Drop(x): zero a whole modality (RGB or Thermal), per sample.

        Returns (dropped_inputs, availability) so the backbone can also gate the
        missing modality's tokens explicitly."""
        B = inputs.shape[0]
        dropped = inputs.clone()
        avail_rgb = torch.ones(B, device=inputs.device)
        avail_thr = torch.ones(B, device=inputs.device)
        drop_rgb = torch.rand(B, device=inputs.device) < 0.5
        for b in range(B):
            if drop_rgb[b]:
                dropped[b, :3] = 0
                avail_rgb[b] = 0.0
            else:
                dropped[b, 3:6] = 0
                avail_thr[b] = 0.0
        return dropped, {'rgb': avail_rgb, 'thermal': avail_thr}

    def loss(self, inputs, data_samples):
        losses = dict()
        # clean view
        losses.update(self._seg_loss(inputs, data_samples, 'clean'))
        # degraded view C(x)
        if self.lambda_deg > 0:
            deg_inputs = self._make_degraded(inputs)
            for k, v in self._seg_loss(deg_inputs, data_samples, 'deg').items():
                losses[k] = self.lambda_deg * v
        # missing-modality view Drop(x)
        if self.lambda_missing > 0:
            miss_inputs, avail = self._make_missing(inputs)
            for k, v in self._seg_loss(miss_inputs, data_samples, 'missing',
                                       availability=avail).items():
                losses[k] = self.lambda_missing * v
        return losses
