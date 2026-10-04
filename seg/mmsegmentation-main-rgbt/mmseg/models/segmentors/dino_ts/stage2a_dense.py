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
from ..unified_degradation import RGBTDegrader


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
        self.degrader = RGBTDegrader(
            mean=self.data_preprocessor.mean.flatten(),
            std=self.data_preprocessor.std.flatten(), **(degradation or {}))
        self.current_epoch = 0

    def _seg_loss(self, inputs, data_samples, prefix, availability=None):
        feats = self.extract_feat(inputs, availability=availability)
        losses = self.decode_head.loss(feats, data_samples, self.train_cfg)
        # rename decode.* -> <prefix>.decode.* to keep the three views distinct
        return {f'{prefix}.{k}': v for k, v in losses.items()}

    @torch.no_grad()
    def _make_degraded(self, inputs, data_samples=None):
        """C(x): sample a recipe from the shared raw-pixel corruption library."""
        rgb, thermal = self._split(inputs)
        shapes = [ds.img_shape for ds in data_samples] if data_samples is not None else None
        drgb, dthr, _, _ = self.degrader(
            rgb, thermal, epoch=self.current_epoch, valid_shapes=shapes)
        return torch.cat([drgb, dthr], dim=1)

    @torch.no_grad()
    def _make_missing(self, inputs, data_samples=None):
        """Drop(x): raw black pixels, identical to the offline failure cases.

        Do not provide privileged availability flags that are absent at test.
        """
        shapes = [ds.img_shape for ds in data_samples] if data_samples is not None else None
        return self.degrader.missing(inputs, valid_shapes=shapes), None

    def loss(self, inputs, data_samples):
        losses = dict()
        # clean view
        losses.update(self._seg_loss(inputs, data_samples, 'clean'))
        # degraded view C(x)
        if self.lambda_deg > 0:
            deg_inputs = self._make_degraded(inputs, data_samples)
            for k, v in self._seg_loss(deg_inputs, data_samples, 'deg').items():
                losses[k] = self.lambda_deg * v
        # missing-modality view Drop(x)
        if self.lambda_missing > 0:
            miss_inputs, avail = self._make_missing(inputs, data_samples)
            for k, v in self._seg_loss(miss_inputs, data_samples, 'missing',
                                       availability=avail).items():
                losses[k] = self.lambda_missing * v
        return losses
