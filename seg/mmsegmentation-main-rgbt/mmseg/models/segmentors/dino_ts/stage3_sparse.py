"""Stage 3: Dense-to-Sparse Distillation (doc section 9).

    Sparse Student <- Frozen Dense Robust Teacher weights   (doc 9.1)

The teacher is fully frozen and keeps ALL extras (dense). The student adds the
Utility Router and prunes extras to a fixed budget K via a REAL gather (doc 9.2:
"正式推理必须通过 gather 缩短序列, 不能只将未选择 Token 乘零"). Early training uses
a soft gate (straight-through), late training and all eval use sparse_hard.

Losses:
  L_compression (doc 9.3): a_sparse(C(x)) vs sg a_dense(C(x))  — same degraded
      input to both; isolates the token-compression error, trains the router.
  L_robust      (doc 9.4): a_sparse(C(x)) vs sg a_dense(x)     — student on
      degraded should match dense teacher on clean.
  L_logit       (doc 9.5): labeled-only KL(teacher_logits, student_logits).
  L_seg         : standard Mask2Former loss on labeled data.

  labeled   = L_seg + lc*L_compression + lr*L_robust + ll*L_logit
  unlabeled = lc*L_compression + lr*L_robust                       (doc 9.6)

Progressive budget (doc 9.7): K ramps N -> 0.75N -> 0.5N -> target_K over
warmup epochs; final eval fixes target_K. Set via a hook or the epoch-driven
schedule below.
"""
from typing import Optional

import torch

from mmseg.registry import MODELS
from mmengine.runner import load_checkpoint
from mmengine.logging import print_log
from .stage2a_dense import DinoTSDense
from . import losses as L


@MODELS.register_module()
class DinoTSSparse(DinoTSDense):
    def __init__(self, *args,
                 teacher_cfg=None,
                 teacher_ckpt=None,
                 init_from_teacher: bool = True,
                 target_k: Optional[int] = None,
                 budget_schedule=(1.0, 0.75, 0.5),  # fractions before target_k
                 budget_warmup_epochs: int = 30,
                 soft_to_hard_epoch: int = 30,       # switch soft->hard gate
                 lambda_comp: float = 1.0,
                 lambda_rob: float = 1.0,
                 lambda_logit: float = 1.0,
                 logit_temperature: float = 2.0,
                 **kwargs):
        # student trains with a differentiable gate early; eval uses sparse_hard
        kwargs.setdefault('forward_mode', 'sparse_hard')
        kwargs['target_k'] = target_k
        super().__init__(*args, **kwargs)
        self.target_k = target_k
        self.budget_schedule = list(budget_schedule)
        self.budget_warmup_epochs = budget_warmup_epochs
        self.soft_to_hard_epoch = soft_to_hard_epoch
        self.lambda_comp = lambda_comp
        self.lambda_rob = lambda_rob
        self.lambda_logit = lambda_logit
        self.logit_temperature = logit_temperature

        # frozen dense teacher (same arch, keeps all extras)
        self.teacher_ckpt = teacher_ckpt
        self.init_from_teacher = init_from_teacher
        self.teacher = None
        if teacher_cfg is not None:
            self.teacher = MODELS.build(teacher_cfg)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

    def init_weights(self):
        # Runner initializes decoder/neck modules after construction. Load the
        # checkpoint afterwards so their init_weights cannot erase the teacher
        # or the student's warm start. Runner's load_from/resume still runs later.
        if self._is_init:
            return
        super().init_weights()
        if self.teacher is not None and self.teacher_ckpt is not None:
            load_checkpoint(self.teacher, self.teacher_ckpt, map_location='cpu')
            if self.init_from_teacher:
                missing, unexpected = self.load_state_dict(
                    self.teacher.state_dict(), strict=False)
                student_only = [k for k in missing if not k.startswith('teacher.')]
                print_log(
                    f'DinoTSSparse warm-start from teacher: '
                    f'{len(student_only)} student-only params kept own init; '
                    f'{len(unexpected)} unexpected.', logger='current')

    def train(self, mode=True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    # -- budget + gate scheduling ------------------------------------------
    def _current_k(self, N):
        """Progressive budget K given the current epoch (doc 9.7)."""
        if self.target_k is None:
            self.target_k = N // 2  # sensible default
        stages = self.budget_schedule
        if not stages or self.current_epoch >= self.budget_warmup_epochs:
            return self.target_k
        # step through the schedule fractions across the warmup window
        step = self.budget_warmup_epochs / (len(stages) + 1)
        idx = min(int(self.current_epoch / max(step, 1)), len(stages) - 1)
        frac = stages[idx]
        return max(self.target_k, int(round(frac * N)))

    def _student_mode(self):
        return ('sparse_hard' if self.current_epoch >= self.soft_to_hard_epoch
                else 'sparse_soft')

    def _sparse_anchors(self, inputs, mode, k, availability=None):
        rgb, thermal = self._split(inputs)
        out = self.backbone(rgb, thermal, mode=mode, target_k=k,
                            soft_tau=self.soft_tau, availability=availability)
        self._last_backbone_out = out
        self._last_seq_len = out['seq_len']
        return out

    @torch.no_grad()
    def _dense_anchors(self, inputs):
        rgb, thermal = self._split(inputs)
        return self.teacher.backbone(rgb, thermal, mode='dense')['anchors']

    def loss(self, inputs, data_samples):
        assert self.teacher is not None, 'Stage 3 requires a frozen teacher_cfg'
        has_label = bool(getattr(data_samples[0], 'has_label', True)) \
            if len(data_samples) else True

        N = self.backbone.grid[0] * self.backbone.grid[1]
        k = self._current_k(N)
        mode = self._student_mode()

        # same degraded input C(x) to student and teacher (doc 9.3)
        deg_inputs = self._make_degraded(inputs, data_samples)

        # student sparse anchors on C(x); build seg feats from the same forward
        s_out = self._sparse_anchors(deg_inputs, mode, k)
        a_sparse = s_out['anchors']
        student_feats = self._anchor_to_pyramid(s_out['anchor_map'])

        # teacher dense anchors on C(x) and on clean x
        a_dense_deg = self._dense_anchors(deg_inputs)
        a_dense_clean = self._dense_anchors(inputs)

        losses = dict()
        # compression + robust feature distillation
        if self.lambda_comp > 0:
            losses['loss_compression'] = self.lambda_comp * L.compression_loss(
                a_sparse, a_dense_deg)
        if self.lambda_rob > 0:
            losses['loss_robust'] = self.lambda_rob * L.robust_loss(
                a_sparse, a_dense_clean)

        if has_label:
            # segmentation loss on the student's degraded-view feats
            seg = self.decode_head.loss(student_feats, data_samples, self.train_cfg)
            for kk, vv in seg.items():
                losses[f'decode.{kk}'] = vv
            # labeled logit distillation (doc 9.5)
            if self.lambda_logit > 0:
                s_logits = self._feats_to_logits(student_feats, data_samples)
                with torch.no_grad():
                    t_feats = self.teacher.extract_feat(deg_inputs)
                    t_logits = self._feats_to_logits(t_feats, data_samples,
                                                     head=self.teacher.decode_head)
                if s_logits is not None and t_logits is not None:
                    losses['loss_logit'] = self.lambda_logit * L.logit_distill_loss(
                        s_logits, t_logits, temperature=self.logit_temperature)
        return losses

    def _feats_to_logits(self, feats, data_samples, head=None):
        """Permutation-invariant per-pixel class map from a Mask2Former head's
        training forward (cls.softmax x mask.sigmoid), used as distillation
        logits — same trick as QualityDistillStudent._head_class_map. Returns
        [B, num_classes, h, w] or None if the head shape is unexpected."""
        head = head or self.decode_head
        all_cls, all_mask = head(feats, data_samples)
        cls_score = all_cls[-1].float().softmax(dim=-1)[..., :-1]  # drop no-object
        mask_pred = all_mask[-1].float().sigmoid()
        return torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)
