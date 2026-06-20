"""Student model: same dual-branch SHARED-Swin Mask2Former as the teacher, PLUS
the quality + cross-modal compensation modules and quality-aware attention bias.
Trained on DEGRADED RGB-T with:
  - segmentation loss (vs GT)
  - feature distillation (per-stage fused feats vs frozen clean teacher, gated)
  - output distillation (final seg logits vs frozen clean teacher)
  - quality supervision (masked BCE + cross-modal ranking, using degrade masks)

Attention bias: ported from swin_quality_mask2former (window-aligned, cyclic
shift handled). Quality is predicted ONCE per stage (on stage-input tokens) and
used BOTH as in-stage attention bias AND for cross-modal compensation at the
stage output. No hard masking (continuous quality, fully differentiable).
Degradation is applied INTERNALLY (Paradigm One, single forward on degraded).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmengine.runner import load_checkpoint
from mmengine.logging import print_log
from .quality_distill_teacher import QualityDistillTeacher
from .quality_distill_modules import (
    StageQuality, CrossModalCompensation, quality_to_swin_bias_convex)
from .degradation import DegradationGenerator
from .swin_quality_mask2former import _replace_swin_blocks_with_quality


@MODELS.register_module()
class QualityDistillStudent(QualityDistillTeacher):
    def __init__(self, *args,
                 teacher_cfg=None,
                 teacher_ckpt=None,
                 init_from_teacher=True,   # warm-start student backbone/fusion/head
                 fusion_dims=(96, 192, 384, 768),
                 use_quality=True,         # E2: full mechanism. False -> E0b/E1.
                 quality_loss_weight=1.0,
                 distill_loss_weight=1.0,
                 output_distill_weight=1.0,
                 bias_alpha=4.0,
                 bias_gamma=3.0,
                 degradation=None,
                 **kwargs):
        super().__init__(*args, fusion_dims=fusion_dims, **kwargs)
        self.use_quality = use_quality

        # quality-aware attention: replace Swin blocks so they accept quality_bias
        _replace_swin_blocks_with_quality(self.backbone)
        self.window_size = self.backbone.stages[0].blocks[0].attn.window_size

        # per-stage quality predictor (shared across modalities) + compensation
        self.quality = nn.ModuleList([StageQuality(c) for c in fusion_dims])
        self.compensate = nn.ModuleList(
            [CrossModalCompensation(c) for c in fusion_dims])

        self.quality_loss_weight = quality_loss_weight
        self.distill_loss_weight = distill_loss_weight
        self.output_distill_weight = output_distill_weight
        self.bias_alpha = bias_alpha
        self.bias_gamma = bias_gamma
        self.current_epoch = 0

        deg = degradation or {}
        self.degrader = DegradationGenerator(**deg)

        # frozen teacher (same arch, clean input) for distillation
        self.teacher = None
        if teacher_cfg is not None:
            self.teacher = MODELS.build(teacher_cfg)
            if teacher_ckpt is not None:
                load_checkpoint(self.teacher, teacher_ckpt, map_location='cpu')
            # warm-start: copy the SHARED params (backbone / fuse_convs / neck /
            # decode head) from the trained teacher into the student itself, as a
            # TRAINABLE starting point. Student-only modules (quality / compensate
            # / quality-aware block extras) keep their own init. strict=False so
            # those non-matching keys are skipped, not errored.
            if init_from_teacher and teacher_ckpt is not None:
                # teacher.state_dict() keys (backbone.* / fuse_convs.* / neck.* /
                # decode_head.*) match the student's own params and get loaded.
                # Student-only modules (quality.* / compensate.*) and the frozen
                # teacher.* copy appear in `missing` and keep their init.
                missing, unexpected = self.load_state_dict(
                    self.teacher.state_dict(), strict=False)
                loaded = sum(1 for _ in self.teacher.state_dict())
                student_only = [k for k in missing
                                if not k.startswith('teacher.')]
                print_log(
                    f'QualityDistillStudent warm-start from teacher: '
                    f'~{loaded} shared params loaded; '
                    f'{len(student_only)} student-only params kept own init; '
                    f'{len(unexpected)} unexpected.', logger='current')
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        # keep the teacher in eval ALWAYS: nn.Module.train() is recursive and
        # would otherwise re-enable the teacher's stochastic depth (drop_path),
        # making the distillation targets random/noisy.
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    # quality-aware feature extraction with attention bias + compensation.
    # Custom Swin forward: predict quality at each stage start (on stage-input
    # tokens), inject it as window attention bias into every block of the stage,
    # then reuse the SAME quality at the stage output for cross-modal compensation.
    def extract_feat(self, inputs):
        B = inputs.shape[0] // 2  # inputs = cat([rgb, t], dim=0), 2B total
        bb = self.backbone

        x, (H, W) = bb.patch_embed(inputs)
        if bb.use_abs_pos_embed:
            x = x + bb.absolute_pos_embed
        x = bb.drop_after_pos(x)
        B_tok = x.shape[0]

        fused_feats, quality_scores = [], []
        rgb_feats, t_feats = [], []  # per-modality (compensated) feats, for vis
        for i, stage in enumerate(bb.stages):
            # quality from stage-input tokens (spatial res constant within stage)
            x_2d = x.reshape(B_tok, H, W, -1).permute(0, 3, 1, 2)
            x_rgb_in, x_t_in = x_2d[:B], x_2d[B:]
            if self.use_quality:
                s_rgb = self.quality[i](x_rgb_in, x_t_in)   # [B,1,H,W]
                s_t = self.quality[i](x_t_in, x_rgb_in)
                # relative (competitive) quality for the attention bias.
                s_avg = (s_rgb + s_t) / 2 + 1e-8
                rel = torch.cat([s_rgb / s_avg, s_t / s_avg], dim=0)
                for block in stage.blocks:
                    shift = block.attn.shift_size
                    qbias = quality_to_swin_bias_convex(
                        rel, self.window_size, shift,
                        alpha=self.bias_alpha, gamma=self.bias_gamma)
                    x = block(x, (H, W), quality_bias=qbias)
            else:
                # mechanism OFF (E0b/E1): no bias, quality forced to 1.
                for block in stage.blocks:
                    x = block(x, (H, W), quality_bias=None)
                ones = x.new_ones(B, 1, H, W)
                s_rgb = s_t = ones

            # stage output feature -> split, compensate, fuse
            if i in bb.out_indices:
                norm_layer = getattr(bb, f'norm{i}', None)
                out = norm_layer(x) if norm_layer is not None else x
                out = out.view(B_tok, H, W, -1).permute(0, 3, 1, 2).contiguous()
                x_rgb, x_t = out[:B], out[B:]
                if self.use_quality:
                    x_rgb_c = self.compensate[i](x_rgb, x_t, s_rgb)
                    x_t_c = self.compensate[i](x_t, x_rgb, s_t)
                    x_rgb, x_t = x_rgb_c, x_t_c
                fused_feats.append(self._fuse_stage(i, x_rgb, x_t))
                quality_scores.append((s_rgb, s_t))
                rgb_feats.append(x_rgb)
                t_feats.append(x_t)

            if stage.downsample is not None:
                x, (H, W) = stage.downsample(x, (H, W))

        self._last_fused_feats = fused_feats
        self._last_quality = quality_scores
        self._last_rgb_feats = rgb_feats
        self._last_t_feats = t_feats
        feats = fused_feats
        if self.with_neck:
            feats = self.neck(fused_feats)
        return feats

    @torch.no_grad()
    def extract_feat_vis(self, inputs):
        """Visualization hook entry. inputs: 6ch RGB-T OR batch-doubled cat.
        Returns per-modality feats, final fused feats, quality maps -- for BOTH
        the clean input and an internally-degraded copy. No private branch
        (this model has none): just RGB feats | T feats | fused feats."""
        if inputs.shape[1] == 6:
            rgb, t = inputs[:, :3], inputs[:, 3:]
        else:
            B = inputs.shape[0] // 2
            rgb, t = inputs[:B], inputs[B:]

        def _run(r, x):
            self.extract_feat(torch.cat([r, x], dim=0))
            qs = self._last_quality
            return (list(self._last_rgb_feats), list(self._last_t_feats),
                    list(self._last_fused_feats),
                    [q[0] for q in qs], [q[1] for q in qs])

        zc_rgb, zc_t, fused, q_rgb, q_t = _run(rgb, t)

        drgb, dt, _, _ = self.degrader(rgb, t, epoch=self.current_epoch)
        (zc_rgb_d, zc_t_d, fused_d, q_rgb_d, q_t_d) = _run(drgb, dt)

        out = dict(
            zc_rgb=zc_rgb, zc_t=zc_t, final_fused=fused,
            clean_rgb_img=rgb, clean_t_img=t,
            zc_rgb_deg=zc_rgb_d, zc_t_deg=zc_t_d, final_fused_deg=fused_d,
            deg_rgb_img=drgb, deg_t_img=dt,
            deg_type_rgb='degraded', deg_type_t='degraded')
        # quality is meaningful only when the mechanism is ON. With use_quality
        # =False (E0b/E1) the scores are forced to 1 -> an all-red, info-less
        # map. Omit the quality keys so the hook skips the quality grid.
        if self.use_quality:
            out.update(q_rgb_maps=q_rgb, q_t_maps=q_t,
                       q_rgb_deg=q_rgb_d, q_t_deg=q_t_d)
        return out

    def predict_with_missing(self, inputs, data_samples=None,
                             mask_rgb=False, mask_t=False):
        """Predict with one modality zeroed (whole-modality missing).

        RGB = channels 0:3, T = channels 3:6 (matches MissingModalityEvalHook).
        Reuses the inherited predict() so postprocess is identical to the
        clean path; extract_feat re-cats to the batch-doubled dual stream.
        """
        inputs = inputs.clone()
        if mask_rgb:
            inputs[:, :3] = 0
        if mask_t:
            inputs[:, 3:] = 0
        return self.predict(inputs, data_samples)

    # ---- quality supervision: 0/1 BCE (missing is binary -> zeroed=0, kept=1).
    # No cross-modal ranking: "degraded" != "lower quality than the other
    # modality" (the other modality's intrinsic quality is unknown), so a
    # cross-modal ordering constraint can be wrong. Absolute per-modality BCE +
    # the downstream seg loss handle "which modality to trust per location".
    def _quality_losses(self, quality_scores, mask_rgb, mask_t):
        bce_total = 0.0
        n = 0
        for (s_rgb, s_t) in quality_scores:
            h, w = s_rgb.shape[-2:]
            m_rgb = F.adaptive_max_pool2d(mask_rgb, (h, w))  # 1 = missing
            m_t = F.adaptive_max_pool2d(mask_t, (h, w))
            for s, m in ((s_rgb, m_rgb), (s_t, m_t)):
                target = (m < 0.5).float()  # kept=1, missing=0
                bce_total = bce_total + F.binary_cross_entropy(s, target)
            n += 1
        return bce_total / max(n, 1)

    # ---- feature distillation: per-stage fused feats, quality-gated
    def _feat_distill_loss(self, student_feats, teacher_feats, quality_scores):
        total = 0.0
        n = 0
        for sf, tf, (s_rgb, s_t) in zip(student_feats, teacher_feats, quality_scores):
            gate = torch.maximum(s_rgb, s_t)  # [B,1,H,W] high where any modality good
            diff = (sf - tf.detach()) ** 2
            total = total + (gate * diff).mean()
            n += 1
        return total / max(n, 1)

    # ---- output distillation: PERMUTATION-INVARIANT per-pixel semantic map.
    # NOTE: per-query mask logits can't be distilled directly -- Mask2Former's
    # Hungarian matching makes query order arbitrary, so teacher query i != student
    # query i. The einsum'd per-pixel class map (cls.softmax x mask.sigmoid) IS
    # permutation-invariant. We build it from the TRAINING forward's raw query
    # outputs (NOT decode_head.predict), normalize to a per-pixel class
    # distribution, and distill with quality-GATED KL.
    #
    # IMPORTANT: do NOT use decode_head.predict() here. predict() re-runs the head
    # AND resizes mask logits to batch_img_metas['pad_shape'] -- which at training
    # time is the AUGMENTED random crop size. Distilling against that pulls the
    # head toward a train-metainfo-specific solution that is mis-scaled at val
    # time (observed: val aAcc ~1%, class indices scrambled, while train loss
    # looks healthy). Building the class map from the head's own (cls, mask)
    # forward outputs at their native feature resolution is metainfo-free and
    # matches the DINOv3 EoMT path.
    def _head_class_map(self, head, feats, data_samples):
        """Permutation-invariant per-pixel class map from a Mask2Former head's
        TRAINING forward. Returns [B, num_classes, h, w] at the mask logits'
        native resolution.

        Uses head(feats, data_samples) -- the SAME forward the training loss
        uses -- NOT predict(). predict() resizes mask logits to
        batch_img_metas['pad_shape'] (the augmented train crop size at train
        time), which corrupts the distillation target and scrambles val-time
        class indices (val aAcc ~1% while train loss looks fine). The raw
        forward outputs are at the head's native mask resolution, metainfo only
        affects the internal padding mask, not the output scale."""
        all_cls, all_mask = head(feats, data_samples)
        cls_score = all_cls[-1].float().softmax(dim=-1)[..., :-1]  # drop no-object
        mask_pred = all_mask[-1].float().sigmoid()
        return torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)

    def _output_distill_loss(self, s_prob, t_prob, gate, eps=1e-6):
        # align teacher map to student resolution if they differ (both are at
        # native mask resolution; same arch -> normally identical, guard anyway)
        if t_prob.shape[-2:] != s_prob.shape[-2:]:
            t_prob = F.interpolate(t_prob, size=s_prob.shape[-2:],
                                   mode='bilinear', align_corners=False)
        s = s_prob / (s_prob.sum(1, keepdim=True) + eps)
        t = t_prob / (t_prob.sum(1, keepdim=True) + eps)
        kl = (t * ((t + eps).log() - (s + eps).log())).sum(1, keepdim=True)
        return (gate * kl).mean()

    def loss(self, inputs, data_samples):
        # Paradigm One: degrade internally, single forward on degraded input
        rgb, t = inputs[:, :3], inputs[:, 3:]
        drgb, dir_, mask_rgb, mask_t = self.degrader(rgb, t, epoch=self.current_epoch)

        rgbt = torch.cat([drgb, dir_], dim=0)
        fused_feats = self.extract_feat(rgbt)
        quality_scores = self._last_quality
        student_fused = self._last_fused_feats

        losses = dict()
        losses.update(self._decode_head_forward_train(fused_feats, data_samples))

        # quality supervision (0/1 BCE; ranking removed)
        if self.use_quality and self.quality_loss_weight > 0:
            bce = self._quality_losses(quality_scores, mask_rgb, mask_t)
            losses['loss_quality_bce'] = self.quality_loss_weight * bce

        # distillation vs frozen clean teacher (feature + output)
        if self.teacher is not None and (
                self.distill_loss_weight > 0 or self.output_distill_weight > 0):
            clean_input = torch.cat([rgb, t], dim=1)
            with torch.no_grad():
                teacher_fused = self.teacher.extract_fused_for_distill(clean_input)
            if self.distill_loss_weight > 0:
                losses['loss_distill_feat'] = self.distill_loss_weight * \
                    self._feat_distill_loss(student_fused, teacher_fused, quality_scores)
            if self.output_distill_weight > 0:
                # build per-pixel class maps from the TRAINING forward (no
                # predict(), no pad_shape resize -- see _head_class_map).
                student_prob = self._head_class_map(
                    self.decode_head, fused_feats, data_samples)
                with torch.no_grad():
                    t_feats = teacher_fused
                    if self.teacher.with_neck:
                        t_feats = self.teacher.neck(teacher_fused)
                    teacher_prob = self._head_class_map(
                        self.teacher.decode_head, t_feats, data_samples)
                # quality gate at output (mask) resolution: relax where both weak
                s_rgb0, s_t0 = quality_scores[0]
                gate = torch.maximum(s_rgb0, s_t0)
                gate = F.interpolate(gate, size=student_prob.shape[-2:],
                                     mode='bilinear', align_corners=False)
                losses['loss_distill_out'] = self.output_distill_weight * \
                    self._output_distill_loss(student_prob, teacher_prob, gate)

        return losses


