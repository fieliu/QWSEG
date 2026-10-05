"""Stage 2B: Dense EMA Robust Self-Distillation (doc section 8).

Initialized from a Stage-2A model that passed acceptance (doc 7.5). An EMA copy
of the online dense model provides stable feature targets on unlabeled data; no
semantic pseudo-labels are ever generated (doc 8.5).

    theta_ema <- momentum * theta_ema + (1 - momentum) * theta_online
    momentum ramps 0.996 -> 0.9999.

Input views (doc 8.2): EMA teacher sees the WEAK (light) view x_weak; the online
model sees the STRONG (heavy) view x_strong. The teacher gets no degradation
encoding; it gains robustness only through EMA of the degradation-supervised
online model.

Losses:
  labeled   = L_stage2A + lambda_anchor * L_anchor (+ lambda_global * L_global_dino)
  unlabeled = lambda_anchor * L_anchor (+ lambda_global * L_global_dino)
L_anchor (doc 8.3): per-position cosine between online anchors (x_strong) and
stop-grad EMA anchors (x_weak).

Batches carry a boolean 'has_label' (default True). Unlabeled batches skip the
segmentation terms. The best EMA weights on val become the Frozen Dense Robust
Teacher for Stage 3 (doc 8.6).
"""
import copy

import torch

from mmseg.registry import MODELS
from mmengine.logging import print_log
from .stage2a_dense import DinoTSDense
from . import losses as L


@MODELS.register_module()
class DinoTSDenseEMA(DinoTSDense):
    def __init__(self, *args,
                 ema_momentum_base: float = 0.996,
                 ema_momentum_final: float = 0.9999,
                 total_epochs: int = 200,
                 lambda_anchor: float = 1.0,
                 lambda_global: float = 0.0,   # optional DINO-style global CE
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_momentum_base = ema_momentum_base
        self.ema_momentum_final = ema_momentum_final
        self.total_epochs = total_epochs
        self.lambda_anchor = lambda_anchor
        self.lambda_global = lambda_global
        self.ema = None
        self.register_buffer(
            '_ema_initialized', torch.tensor(False, dtype=torch.bool))
        # Register the teacher before MMDistributedDataParallel is constructed.
        # Its weights are synchronized from the checkpoint-loaded online model
        # by EMAUpdateHook.before_train().
        self.ema = copy.deepcopy(self)
        self.ema.ema = None  # no recursive EMA-of-EMA
        for p in self.ema.parameters():
            p.requires_grad = False
        self.ema.eval()

    # -- EMA teacher management --------------------------------------------
    @torch.no_grad()
    def initialize_ema(self):
        """Copy checkpoint-loaded online weights into the registered teacher."""
        if bool(self._ema_initialized.item()):
            return
        online = {
            k: v for k, v in self.state_dict().items()
            if not k.startswith('ema.') and k != '_ema_initialized'
        }
        self.ema.load_state_dict(online, strict=False)
        self.ema.eval()
        self._ema_initialized.fill_(True)
        print_log('DinoTSDenseEMA: initialized registered EMA teacher from '
                  'checkpoint-loaded online weights.',
                  logger='current')

    def _current_momentum(self):
        t = min(max(self.current_epoch, 0), self.total_epochs)
        frac = t / max(self.total_epochs, 1)
        return (self.ema_momentum_base
                + (self.ema_momentum_final - self.ema_momentum_base) * frac)

    @torch.no_grad()
    def update_ema(self):
        """Update after the optimizer step; called by EMAUpdateHook."""
        if not bool(self._ema_initialized.item()):
            raise RuntimeError('EMA teacher must be initialized before updating')
        m = self._current_momentum()
        online_params = dict(self.named_parameters())
        for name, pe in self.ema.named_parameters():
            po = online_params[name]
            pe.mul_(m).add_(po.detach(), alpha=1 - m)
        online_buffers = dict(self.named_buffers())
        for name, be in self.ema.named_buffers():
            if name == '_ema_initialized':
                continue
            bo = online_buffers[name]
            if torch.is_floating_point(be):
                be.mul_(m).add_(bo.detach(), alpha=1 - m)
            else:
                be.copy_(bo)

    def train(self, mode=True):
        super().train(mode)
        if self.ema is not None:
            self.ema.eval()  # teacher always eval (no drop_path noise)
        return self

    # -- anchors from a forward --------------------------------------------
    def _anchors_for(self, inputs, model, mode='dense', availability=None):
        model.extract_feat(inputs, mode=mode, availability=availability)
        return model._last_backbone_out['anchors']

    def loss(self, inputs, data_samples):
        if not bool(self._ema_initialized.item()):
            raise RuntimeError(
                'EMA teacher is not initialized. Add EMAUpdateHook to custom_hooks.')

        has_label = bool(getattr(data_samples[0], 'has_label', True)) \
            if len(data_samples) else True

        # weak (light) / strong (heavy) paired views
        rgb, thermal = self._split(inputs)
        mean = self.data_preprocessor.mean.flatten()
        std = self.data_preprocessor.std.flatten()
        (l_rgb, l_thr, h_rgb, h_thr, *_rest) = self.degrader.make_paired(
            rgb, thermal, mean, std, epoch=self.current_epoch,
            valid_shapes=[ds.img_shape for ds in data_samples])
        x_weak = torch.cat([l_rgb, l_thr], dim=1)
        x_strong = torch.cat([h_rgb, h_thr], dim=1)

        losses = dict()

        # segmentation (labeled only): reuse Stage-2A three-view seg on x_strong
        if has_label:
            losses.update(self._seg_loss(inputs, data_samples, 'clean'))
            deg_inputs = self._make_degraded(inputs, data_samples)
            for k, v in self._seg_loss(deg_inputs, data_samples, 'deg').items():
                losses[k] = self.lambda_deg * v
            miss_inputs, avail = self._make_missing(inputs, data_samples)
            for k, v in self._seg_loss(miss_inputs, data_samples, 'missing',
                                       availability=avail).items():
                losses[k] = self.lambda_missing * v

        # anchor consistency: online(x_strong) vs stop-grad EMA(x_weak)
        a_online = self._anchors_for(x_strong, self)
        with torch.no_grad():
            a_ema = self._anchors_for(x_weak, self.ema)
        losses['loss_anchor'] = self.lambda_anchor * L.anchor_consistency_loss(
            a_online, a_ema.detach())

        return losses

    def predict(self, inputs, data_samples=None):
        """Validate the EMA teacher, which is the artifact consumed by Stage 3."""
        if bool(self._ema_initialized.item()):
            return self.ema.predict(inputs, data_samples)
        return super().predict(inputs, data_samples)
