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

        if self.degrader.weak_severity != 0:
            raise NotImplementedError(
                'The Stage-2B objective assumes weak_severity=0, so the EMA '
                'target is the clean input we already hold.  A non-zero '
                'weak_severity needs its own severity-1 teacher pass; add it '
                'explicitly rather than silently degrading the target.')

        has_label = (bool(getattr(data_samples[0], 'has_label', True))
                     if len(data_samples) else True)

        losses = dict()

        # L_stage2A (doc 7.4): supervised segmentation on the CLEAN input.
        if has_label:
            losses.update(self._seg_loss(inputs, data_samples, 'clean'))

        # ONE stochastic strong view per sample (doc 8.2).  This is the single
        # place the shared degradation policy is consulted, so degrade_prob,
        # modality_probs, scope_probs, severity_range and missing_prob all take
        # effect exactly as the policy intends: one view per sample, sampled
        # once.  Drawing several independent views per step (as the previous
        # implementation did) multiplied the cost and made those probabilities
        # meaningless, because the segmentation loss and the anchor loss ended
        # up looking at two different random corruptions of the same batch.
        x_strong = self._make_degraded(inputs, data_samples)

        # Strong-view anchors.  With lambda_deg > 0 the strong view also carries
        # the supervised segmentation loss (design C), so its forward is run
        # ONCE here and the anchors are read straight out of that same pass --
        # extract_feat stashes them in _last_backbone_out.  With lambda_deg = 0
        # there is no segmentation term and the anchors are taken directly
        # (design A).  Either way the strong view is forwarded exactly once.
        if has_label and self.lambda_deg > 0:
            for k, v in self._seg_loss(x_strong, data_samples, 'deg').items():
                losses[k] = self.lambda_deg * v
            a_online = self._last_backbone_out['anchors']
        else:
            a_online = self._anchors_for(x_strong, self)

        # L_anchor (doc 8.3): pull the online anchors produced from the strong
        # view towards the stop-grad EMA anchors produced from the clean view.
        with torch.no_grad():
            a_ema = self._anchors_for(inputs, self.ema)
        losses['loss_anchor'] = self.lambda_anchor * L.anchor_consistency_loss(
            a_online, a_ema.detach())

        return losses

    def predict(self, inputs, data_samples=None):
        """Validate the EMA teacher, which is the artifact consumed by Stage 3."""
        if bool(self._ema_initialized.item()):
            return self.ema.predict(inputs, data_samples)
        return super().predict(inputs, data_samples)
