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
from .distill_utils import prepare_ssl_outputs


@MODELS.register_module()
class QualityDistillStudent(QualityDistillTeacher):
    def __init__(self, *args,
                 teacher_cfg=None,
                 teacher_ckpt=None,
                 init_from_teacher=True,   # warm-start student backbone/fusion/head
                 fusion_dims=(96, 192, 384, 768),
                 use_quality=True,         # E2: full mechanism. False -> E0b/E1.
                 use_compensation=True,    # cross-modal compensation (slaved to use_quality)
                 quality_loss_weight=1.0,
                 distill_loss_weight=1.0,
                 output_distill_weight=1.0,
                 distill_temperature=4.0,
                 clean_floor_weight=0.1,
                 clean_floor=0.5,
                 bias_alpha=4.0,
                 bias_gamma=3.0,
                 freeze_backbone=False,
                 degradation=None,
                 **kwargs):
        super().__init__(*args, fusion_dims=fusion_dims, **kwargs)
        self.use_quality = use_quality
        # compensation is slaved to use_quality (off when mechanism off), and can
        # be turned off independently while keeping the attention bias on (the
        # F1 "bias-only" ablation). Mirrors DINOv3 EoMTRGBTQuality.
        self.use_compensation = use_compensation and use_quality

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
        self.distill_temperature = distill_temperature
        self.clean_floor_weight = clean_floor_weight
        self.clean_floor = clean_floor
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

        # frozen-backbone experiment: freeze the ORIGINAL pretrained Swin
        # backbone (incl. QualitySwinBlock weights copied from the original
        # blocks), train only the increments (quality / compensation / fusion)
        # and the task head. Validates the detachable-increment claim. The
        # quality modules (self.quality / self.compensate) live under separate
        # attributes and stay trainable.
        if freeze_backbone:
            n_frozen = 0
            for p in self.backbone.parameters():
                p.requires_grad = False
                n_frozen += 1
            print_log(
                f'QualityDistillStudent: freeze_backbone=True -> froze '
                f'{n_frozen} backbone param tensors; quality/fusion/decode-head '
                f'trainable.', logger='current')

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
                # match the per-modality batch B (x_rgb_in/x_t_in are x_2d[:B]/
                # [B:], each B). B == per-modality count for both the single path
                # (B=B_orig) and the paired path (B=2*B_orig); use B, not B_tok
                # (=2B, the full rgb+t stack), so s_rgb/s_t align with x_rgb_in.
                ones = x.new_ones(B, 1, H, W)
                s_rgb = s_t = ones

            # stage output feature -> split, compensate, fuse
            if i in bb.out_indices:
                norm_layer = getattr(bb, f'norm{i}', None)
                out = norm_layer(x) if norm_layer is not None else x
                out = out.view(B_tok, H, W, -1).permute(0, 3, 1, 2).contiguous()
                x_rgb, x_t = out[:B], out[B:]
                if self.use_compensation:
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

        E2 (use_quality + make_paired): visualizes the TWO versions the model is
        ACTUALLY trained on -- LIGHT (group 1) and HEAVY (group 2) of the paired
        multi-level degradation. There is NO clean version in paired training,
        so group 1 is the LIGHT degraded input (not the original image). The two
        quality maps side by side show the rank effect (same location: light
        quality > heavy quality).

        E0b/E1 (use_quality=False): keeps the old clean-vs-single-degrade view.
        No private branch: just RGB feats | T feats | fused feats."""
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

        if self.use_quality and hasattr(self.degrader, 'make_paired'):
            # paired training: group1 = LIGHT, group2 = HEAVY (both are real
            # training inputs; no clean pass exists in this regime).
            mean = self.data_preprocessor.mean.flatten()
            std = self.data_preprocessor.std.flatten()
            (l_rgb, l_ir, h_rgb, h_ir, _rr, _ri) = self.degrader.make_paired(
                rgb, t, mean, std, epoch=self.current_epoch)
            g1_rgb, g1_t = l_rgb, l_ir   # light
            g2_rgb, g2_t = h_rgb, h_ir   # heavy
            tag1, tag2 = 'light', 'heavy'
        else:
            # E0b/E1: group1 = clean original, group2 = single-level degraded.
            g1_rgb, g1_t = rgb, t
            g2_rgb, g2_t, _, _ = self.degrader(rgb, t, epoch=self.current_epoch)
            tag1, tag2 = 'clean', 'degraded'

        zc_rgb, zc_t, fused, q_rgb, q_t = _run(g1_rgb, g1_t)
        (zc_rgb_d, zc_t_d, fused_d, q_rgb_d, q_t_d) = _run(g2_rgb, g2_t)

        out = dict(
            zc_rgb=zc_rgb, zc_t=zc_t, final_fused=fused,
            clean_rgb_img=g1_rgb, clean_t_img=g1_t,
            zc_rgb_deg=zc_rgb_d, zc_t_deg=zc_t_d, final_fused_deg=fused_d,
            deg_rgb_img=g2_rgb, deg_t_img=g2_t,
            deg_type_rgb=tag1, deg_type_t=tag2)
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
        """DEPRECATED (not called). Old binary BCE quality supervision; replaced
        by _rank_loss_spatial (paired light/heavy ranking) because binary
        supervision collapsed the quality score to a 2-valued detector. Kept for
        a potential BCE-vs-rank ablation."""
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
            # L2-normalize per-pixel feature vectors over channels (dim=1) before
            # MSE -> direction (cosine-like) match, not dominated by scale.
            sfn = F.normalize(sf, dim=1)
            tfn = F.normalize(tf.detach(), dim=1)
            diff = (sfn - tfn) ** 2
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
        # normalize to proper distributions, temperature-soften via p^(1/T)
        # (these are einsum prob maps, not logits, so softmax(logits/T) doesn't
        # apply; p^(1/T) flattens the per-pixel distribution to surface dark
        # knowledge), then KL (NO T^2 scaling -- see below).
        T = self.distill_temperature
        s = s_prob.clamp_min(0) + eps
        t = t_prob.clamp_min(0) + eps
        s = s / s.sum(1, keepdim=True)
        t = t / t.sum(1, keepdim=True)
        if T != 1.0:
            s = s ** (1.0 / T); s = s / s.sum(1, keepdim=True)
            t = t ** (1.0 / T); t = t / t.sum(1, keepdim=True)
        kl = (t * (t.log() - s.log())).sum(1, keepdim=True)  # >= 0
        # NO T^2 scaling: these are PROBABILITY maps (einsum of cls.softmax x
        # mask.sigmoid), not raw logits. p^(1/T) softening does shrink gradients
        # by ~1/T, but T^2 over-corrects 16x (T=4) and lets the dense per-pixel
        # distill gradient dominate the sparse seg gradient, destroying query
        # specialization (E1 collapse: unlabeled IoU=0, aAcc=5.5%). DINOv3
        # never used T^2 here and works fine. Drop it for cross-backbone parity.
        return (gate * kl.clamp_min(0)).mean()

    def loss(self, inputs, data_samples):
        # E2 (use_quality=True): PAIRED multi-level degradation for quality
        # ranking (mirrors DINOv3 EoMTRGBTQuality). E0b/E1 (use_quality=False):
        # single-level path (unchanged).
        if self.use_quality:
            return self._loss_paired(inputs, data_samples)
        return self._loss_single(inputs, data_samples)

    def _add_distill(self, losses, rgb, t, fused_feats, student_fused,
                     quality_scores, data_samples):
        """Feature + output distillation vs frozen clean teacher. Shared by the
        single and paired paths; all args here are the LIGHT (or only) version."""
        if self.teacher is None or (
                self.distill_loss_weight <= 0 and self.output_distill_weight <= 0):
            return
        clean_input = torch.cat([rgb, t], dim=1)
        with torch.no_grad():
            teacher_fused = self.teacher.extract_fused_for_distill(clean_input)
        if self.distill_loss_weight > 0:
            losses['loss_distill_feat'] = self.distill_loss_weight * \
                self._feat_distill_loss(student_fused, teacher_fused, quality_scores)
        if self.output_distill_weight > 0:
            # Standard Mask2Former distillation (GuidedDistillation): teacher's
            # confident-foreground queries -> pseudo-GT instances, fed to the
            # head's OWN standard loss (Hungarian match + class CE + mask BCE +
            # dice). Reuses the exact standard loss — no hand-rolled matching /
            # full-image BCE (which collapsed foreground mIoU).
            all_cls, all_mask = self.decode_head(fused_feats, data_samples)
            with torch.no_grad():
                t_feats = teacher_fused
                if self.teacher.with_neck:
                    t_feats = self.teacher.neck(teacher_fused)
                t_all_cls, t_all_mask = self.teacher.decode_head(
                    t_feats, data_samples)
                pseudo = prepare_ssl_outputs(
                    t_all_cls[-1].detach(), t_all_mask[-1].detach())
            batch_gt, batch_metas = self._pseudo_to_instances(
                pseudo, data_samples)
            d = self.decode_head.loss_by_feat(
                all_cls, all_mask, batch_gt, batch_metas)
            losses['loss_distill_out'] = self.output_distill_weight * sum(
                v for v in d.values())

    def _pseudo_to_instances(self, pseudo, data_samples):
        """pseudo [{labels,masks}] -> mmdet (batch_gt_instances, batch_img_metas)
        for decode_head.loss_by_feat. masks are bool [K,h,w] at mask resolution;
        mmdet's loss point-samples them, so resolution need not match input."""
        from mmengine.structures import InstanceData
        batch_gt, batch_metas = [], []
        for i, p in enumerate(pseudo):
            inst = InstanceData()
            inst.labels = p['labels'].long()
            inst.masks = p['masks'].float()
            batch_gt.append(inst)
            batch_metas.append(data_samples[i].metainfo)
        return batch_gt, batch_metas

    # ---- E0b/E1 path: single-level degradation, no quality ranking ----
    def _loss_single(self, inputs, data_samples):
        rgb, t = inputs[:, :3], inputs[:, 3:]
        drgb, dir_, mask_rgb, mask_t = self.degrader(rgb, t, epoch=self.current_epoch)
        rgbt = torch.cat([drgb, dir_], dim=0)
        fused_feats = self.extract_feat(rgbt)
        quality_scores = self._last_quality
        student_fused = self._last_fused_feats

        losses = dict()
        losses.update(self._decode_head_forward_train(fused_feats, data_samples))
        self._add_distill(losses, rgb, t, fused_feats, student_fused,
                          quality_scores, data_samples)
        return losses

    # ---- E2 path: paired light/heavy degradation + spatial rank quality loss --
    def _loss_paired(self, inputs, data_samples):
        rgb, t = inputs[:, :3], inputs[:, 3:]
        mean = self.data_preprocessor.mean.flatten()
        std = self.data_preprocessor.std.flatten()
        (l_rgb, l_ir, h_rgb, h_ir, region_rgb, region_ir) = self.degrader.make_paired(
            rgb, t, mean, std, epoch=self.current_epoch)
        B = rgb.shape[0]

        # Swin extract_feat takes cat([rgb, t], 0). Pack light+heavy into batch:
        # cat([l_rgb, h_rgb, l_ir, h_ir], 0) -> inside, B'=2B, x_rgb=[l;h]rgb,
        # x_t=[l;h]ir. Quality scores come out [2B,1,H,W] -> [:B]=light, [B:]=heavy.
        rgbt = torch.cat([l_rgb, h_rgb, l_ir, h_ir], dim=0)
        fused_feats = self.extract_feat(rgbt)        # list of [2B,C,H,W] per stage
        quality_scores = self._last_quality          # list of ([2B,1,H,W],[2B,1,H,W])
        student_fused = self._last_fused_feats

        # segmentation loss on BOTH versions: fused_feats are [2B,...]; the head
        # consumes them with duplicated targets (light + heavy share GT).
        targets2 = list(data_samples) + list(data_samples)
        losses = dict()
        losses.update(self._decode_head_forward_train(fused_feats, targets2))

        # rank-only quality loss: in the degraded modality, s(light) > s(heavy)
        if self.quality_loss_weight > 0:
            losses['loss_quality'] = self.quality_loss_weight * \
                self._rank_loss_spatial(quality_scores, region_rgb, region_ir, B)
            # clean-region soft floor (anti-collapse): non-degraded pixels keep
            # quality >= clean_floor. Soft lower bound, not a hard push to 1.
            if self.clean_floor_weight > 0:
                losses['loss_clean_floor'] = self.clean_floor_weight * \
                    self._clean_floor_spatial(quality_scores, region_rgb, region_ir, B)

        # distillation: align the LIGHT version (slice [:B]) to the clean teacher
        light_fused = [f[:B] for f in fused_feats]
        light_student_fused = [f[:B] for f in student_fused]
        light_q = [(sr[:B], st[:B]) for (sr, st) in quality_scores]
        self._add_distill(losses, rgb, t, light_fused, light_student_fused,
                          light_q, data_samples)
        return losses

    # ---- spatial rank quality loss (Swin: scores are [B,1,H,W] per stage) ----
    def _rank_loss_spatial(self, quality_scores, region_rgb, region_ir, B,
                           margin=0.35):
        """For each stage and each modality, on the DEGRADED modality's region
        push the LIGHT quality above the HEAVY quality by at least `margin`:
            L = mean max(0, margin - (s_light - s_heavy)).
        quality_scores entries are [2B,1,H,W] (light=0:B, heavy=B:2B). Large
        margin prevents the scale-collapse a pure pairwise rank can suffer.
        Only the degraded modality carries signal (the clean modality is
        identical in both versions). No absolute anchor -> absolute level free,
        shaped by the segmentation task. Mirrors DINOv3 _rank_loss."""
        total = z = 0.0
        for (s_rgb, s_t) in quality_scores:
            h, w = s_rgb.shape[-2:]
            for s, reg in ((s_rgb, region_rgb), (s_t, region_ir)):
                reg_d = F.adaptive_max_pool2d(reg, (h, w))   # [B,1,h,w]
                s_light, s_heavy = s[:B], s[B:]              # [B,1,h,w] each
                gap = s_light - s_heavy
                hinge = (margin - gap).clamp_min(0.0)
                denom = reg_d.sum().clamp_min(1.0)
                total = total + (hinge * reg_d).sum() / denom
                z += 1
        return total / max(z, 1)

    def _clean_floor_spatial(self, quality_scores, region_rgb, region_ir, B):
        """Anti-collapse soft floor on NON-degraded pixels (reg==0), spatial
        version. Both light and heavy (full 2B) keep quality >= clean_floor
        there. Soft lower bound relu(floor - s): only penalizes s<floor, leaves
        [floor,1] free. Mirrors DINOv3 _clean_floor_loss."""
        floor = self.clean_floor
        total = z = 0.0
        for (s_rgb, s_t) in quality_scores:
            h, w = s_rgb.shape[-2:]
            for s, reg in ((s_rgb, region_rgb), (s_t, region_ir)):
                reg_d = F.adaptive_max_pool2d(reg, (h, w))   # [B,1,h,w]
                clean = (reg_d < 0.5).float()                # non-degraded
                clean = torch.cat([clean, clean], dim=0)     # [2B,1,h,w]
                below = (floor - s).clamp_min(0.0)
                denom = clean.sum().clamp_min(1.0)
                total = total + (below * clean).sum() / denom
                z += 1
        return total / max(z, 1)


