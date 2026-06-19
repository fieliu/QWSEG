"""Quality-aware multimodal (RGB-T) EoMT.

Deliverable 3 (Paradigm One training):
- Shares the dual-stream + cross-attention skeleton of EoMTRGBTFusion.
- At each fusion point, a SHARED TokenQualityPredictor (with cross-modal
  context) scores each token's quality. The score is used as:
    (a) an additive kv-bias in cross-attention (low-quality tokens of the
        OTHER modality are suppressed as keys -> less contamination), and
    (b) a per-position weight when merging the two streams into one sequence.
- Training uses internal degradation (Paradigm One): the data_preprocessor
  feeds clean RGB-T; this model degrades ONE modality on the fly, feeds the
  degraded input through a SINGLE forward pass, and supervises:
    * segmentation loss (mask-classification), and
    * a masked-BCE quality loss: degraded positions -> target 0 (treated as the
      worst regardless of severity, per the user's request), clean positions ->
      target 1. Intermediate positions are not in either mask -> ignored.

Curriculum and degradation type are controlled internally via the
DegradationGenerator config.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmengine.runner import load_checkpoint
from mmengine.logging import print_log
from .eomt_rgbt_fusion import EoMTRGBTFusion
from .eomt_fusion_blocks import TokenQualityPredictor, TokenCrossModalCompensation
from .degradation import DegradationGenerator
from .eomt_utils import build_targets, mask_class_to_seg_logits, resize_seg_logits
from .eomt_quality_attn import (
    wrap_backbone_attention, quality_score_to_token_bias)


@MODELS.register_module()
class EoMTRGBTQuality(EoMTRGBTFusion):
    def __init__(self, *args,
                 quality_loss_weight=1.0,
                 fuse_tau=0.5,
                 bias_alpha=4.0,
                 bias_gamma=3.0,
                 use_quality=True,
                 use_self_attn_bias=True,
                 use_compensation=True,
                 teacher_cfg=None,
                 teacher_ckpt=None,
                 init_from_teacher=True,
                 distill_loss_weight=1.0,
                 output_distill_weight=1.0,
                 degradation=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        dim = self.network.encoder.backbone.embed_dim

        # Quality is evaluated at SEGMENT boundaries: after embedding (segment 0)
        # and after each fusion point. One predictor per segment boundary.
        self.num_segments = len(self.fusion_points) + 1
        self.quality_predictors = nn.ModuleList(
            [TokenQualityPredictor(dim) for _ in range(self.num_segments)]
        )
        # cross-modal compensation, one per segment boundary (applied before merge
        # uses the last; intermediate ones repair tokens entering the next segment)
        self.compensate = nn.ModuleList(
            [TokenCrossModalCompensation(dim) for _ in range(self.num_segments)]
        )
        self.quality_loss_weight = quality_loss_weight
        self.fuse_tau = fuse_tau
        self.bias_alpha = bias_alpha
        self.bias_gamma = bias_gamma
        # master switch: use_quality=False (E0b/E1) forces quality=1 everywhere,
        # which makes the bias (-a*(1-1)^g=0), the merge (softmax([1,1])=mean) and
        # the compensation (x*1+0=x) all collapse to the EoMTRGBTFusion baseline.
        # The two fine-grained switches are slaved to it so they can't re-enable
        # mechanism pieces when the master is off.
        self.use_quality = use_quality
        self.use_self_attn_bias = use_self_attn_bias and use_quality
        self.use_compensation = use_compensation and use_quality

        # wrap each backbone block's attention so we can inject a pre-softmax
        # quality key-bias into the modality-internal self-attention. Done AFTER
        # the teacher warm-start below would change key names, so the warm-start
        # must run first -> defer wrapping to the end of __init__.

        self.distill_loss_weight = distill_loss_weight
        self.output_distill_weight = output_distill_weight

        # frozen clean teacher (same EoMTRGBTFusion arch) for distillation.
        # Built BEFORE attention wrapping so the warm-start load_state_dict sees
        # matching key names (network...blocks.{i}.attn.* on both sides).
        self.teacher = None
        if teacher_cfg is not None:
            self.teacher = MODELS.build(teacher_cfg)
            if teacher_ckpt is not None:
                load_checkpoint(self.teacher, teacher_ckpt, map_location='cpu')
            # warm-start: copy the SHARED params (backbone / fusions / q / norms)
            # from the trained teacher into the student as a TRAINABLE start.
            # Student-only modules (quality_predictors / compensate) and the
            # frozen teacher.* copy fall into `missing` and keep their own init.
            if init_from_teacher and teacher_ckpt is not None:
                missing, unexpected = self.load_state_dict(
                    self.teacher.state_dict(), strict=False)
                loaded = sum(1 for _ in self.teacher.state_dict())
                student_only = [k for k in missing
                                if not k.startswith('teacher.')]
                print_log(
                    f'EoMTRGBTQuality warm-start from teacher: '
                    f'~{loaded} shared params loaded; '
                    f'{len(student_only)} student-only params kept own init; '
                    f'{len(unexpected)} unexpected.', logger='current')
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        # NOW wrap attention (after warm-start). Wrapping renames attn -> attn.attn
        # internally, but the wrapper delegates so EoMT's _attn still works, and
        # the optimizer's 'network.encoder.backbone' prefix still matches.
        if self.use_self_attn_bias:
            self._attn_wrappers = wrap_backbone_attention(
                self.network.encoder.backbone)

        deg_cfg = degradation or {}
        self.degrader = DegradationGenerator(**deg_cfg)
        # set by TrainVisHook (model.current_epoch = runner.epoch); default 0
        self.current_epoch = 0
        # filled per forward for visualization / merge / supervision
        self._last_quality = None
        self._all_quality = None
        self._last_merged_feat = None

    def train(self, mode=True):
        # keep the teacher in eval ALWAYS: nn.Module.train() is recursive and
        # would otherwise re-enable the teacher's dropout, making distillation
        # targets noisy.
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    # ---- run one block on one stream, optionally injecting a self-attn bias ----
    def _run_block_q(self, block, net, x, rope, kv_bias):
        if self.use_self_attn_bias and kv_bias is not None:
            attn_mod = block.attn if hasattr(block, 'attn') else block.attention
            attn_mod.set_bias(kv_bias)
        return self._run_block(block, net, x, None, rope)

    # ---- segment-based dual-stream forward (overrides parent) ----
    # Quality is evaluated at each segment boundary (after embedding, and after
    # each fusion). The score is reused 3 ways within its segment:
    #   (1) self-attn key-bias for EVERY block in the segment (stop intra-modal
    #       contamination at the source), (2) cross-attn fusion bias at the
    #       fusion point, (3) cross-modal compensation + merge weighting.
    def _dual_stream_forward(self, rgb, t, return_quality=False):
        net = self.network
        backbone = net.encoder.backbone
        num_layers = len(backbone.blocks)

        def embed(x):
            r = backbone.rope_embeddings(x) if hasattr(backbone, "rope_embeddings") else None
            x = backbone.patch_embed(x)
            if hasattr(backbone, "_pos_embed"):
                x = backbone._pos_embed(x)
            return x, r

        z_rgb, rope = embed(rgb)
        z_t, _ = embed(t)

        fp_to_seg = {p: i + 1 for i, p in enumerate(self.fusion_points)}
        all_quality = []  # (s_rgb, s_t) per segment boundary, for vis/supervision
        # per-segment feats for Swin-style 3-col vis: RGB / T (pre-merge) / fused
        vis_rgb, vis_t, vis_fused = [], [], []
        self._fuse_call = 0

        # quality for segment 0 (from embedded tokens)
        seg = 0
        s_rgb, s_t = self._eval_quality(seg, z_rgb, z_t)
        all_quality.append((s_rgb, s_t))
        b_rgb = quality_score_to_token_bias(s_rgb, self.bias_alpha, self.bias_gamma)
        b_t = quality_score_to_token_bias(s_t, self.bias_alpha, self.bias_gamma)

        for i in range(self.decode_start):
            block = backbone.blocks[i]
            if i in fp_to_seg:
                # fuse (cross-attn with convex quality bias), then re-evaluate
                # quality for the NEW segment on the fused tokens.
                z_rgb, z_t = self._fuse_q(z_rgb, z_t, s_rgb, s_t)
                seg = fp_to_seg[i]
                s_rgb, s_t = self._eval_quality(seg, z_rgb, z_t)
                all_quality.append((s_rgb, s_t))
                # record this fusion stage: pre-merge streams + their merge
                # (vis only -- skip during training to avoid extra merge cost)
                if not self.training:
                    vis_rgb.append(z_rgb)
                    vis_t.append(z_t)
                    vis_fused.append(self._merge_q(z_rgb, z_t, s_rgb, s_t, seg))
                b_rgb = quality_score_to_token_bias(s_rgb, self.bias_alpha, self.bias_gamma)
                b_t = quality_score_to_token_bias(s_t, self.bias_alpha, self.bias_gamma)
            # each stream suppresses ITS OWN low-quality tokens as keys:
            z_rgb = self._run_block_q(block, net, z_rgb, rope, b_rgb)
            z_t = self._run_block_q(block, net, z_t, rope, b_t)

        # compensation on the final segment, then quality-weighted merge
        z = self._merge_q(z_rgb, z_t, s_rgb, s_t, seg)

        self._all_quality = all_quality
        self._last_quality = all_quality[-1]
        self._last_merged_feat = z          # [B,N,C], for visualization
        # final stage feats for vis (last fusion's streams + the real merge)
        if not self.training:
            if not vis_rgb:            # no fusion point hit (single-segment): seed
                vis_rgb.append(z_rgb)
                vis_t.append(z_t)
            else:
                vis_fused[-1] = z      # replace last stage's merge with the real one
            self._vis_rgb_feats = vis_rgb
            self._vis_t_feats = vis_t
            self._vis_fused_feats = vis_fused if vis_fused else [z]

        # ---- EoMT query-decode stage (unchanged; merged tokens are clean) ----
        attn_mask = None
        ml_layers, cl_layers = [], []
        x = z
        for i in range(self.decode_start, num_layers):
            block = backbone.blocks[i]
            if i == self.decode_start:
                x = torch.cat(
                    (net.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1)
            if net.masked_attn_enabled:
                ml, cl = net._predict(backbone.norm(x))
                ml_layers.append(ml)
                cl_layers.append(cl)
                attn_mask = net._attn_mask(x, ml, i)
            x = self._run_block(block, net, x, attn_mask, rope)
        ml, cl = net._predict(backbone.norm(x))
        ml_layers.append(ml)
        cl_layers.append(cl)

        if return_quality:
            return ml_layers, cl_layers, all_quality
        return ml_layers, cl_layers

    def _eval_quality(self, seg, z_rgb, z_t):
        # mechanism OFF (E0b/E1): force quality to 1 so the bias collapses to 0,
        # the merge to a plain mean, and compensation to identity -> exactly the
        # EoMTRGBTFusion baseline, with no quality params on the grad path.
        if not self.use_quality:
            ones = z_rgb.new_ones(z_rgb.shape[0], z_rgb.shape[1], 1)
            return ones, ones
        qp = self.quality_predictors[seg]
        return qp(z_rgb, z_t), qp(z_t, z_rgb)   # [B,N,1] each

    def _fuse_q(self, z_rgb, z_t, s_rgb, s_t):
        """Cross-attn fusion with CONVEX quality bias on the key modality."""
        bias_t = quality_score_to_token_bias(s_t, self.bias_alpha, self.bias_gamma)
        bias_rgb = quality_score_to_token_bias(s_rgb, self.bias_alpha, self.bias_gamma)
        fusion = self.fusions[self._fuse_call]
        new_rgb = fusion(z_rgb, z_t, kv_bias=bias_t)   # RGB attends to T; low-q T keys suppressed
        new_t = fusion(z_t, z_rgb, kv_bias=bias_rgb)
        self._fuse_call += 1
        return new_rgb, new_t

    def _merge_q(self, z_rgb, z_t, s_rgb, s_t, seg):
        if self.use_compensation:
            comp = self.compensate[seg]
            z_rgb_c = comp(z_rgb, z_t, s_rgb)
            z_t_c = comp(z_t, z_rgb, s_t)
            z_rgb, z_t = z_rgb_c, z_t_c
        w = torch.softmax(torch.cat([s_rgb, s_t], dim=-1) / self.fuse_tau, dim=-1)
        return w[..., 0:1] * z_rgb + w[..., 1:2] * z_t


    # ---- quality supervision (masked BCE) ----
    def _quality_loss(self, quality_info, mask_rgb_tok, mask_t_tok):
        """For every fusion point's scores, supervise only the EXTREME-certain
        positions: degraded (any severity) -> target 0, clean -> target 1.
        Positions that are neither fully-clean nor degraded are ignored."""
        total = z = 0.0
        for (s_rgb, s_t) in quality_info:
            for s, m in ((s_rgb, mask_rgb_tok), (s_t, mask_t_tok)):
                s = s.squeeze(-1)            # [B, N] (may include prefix tokens)
                # align with the patch-grid mask: drop leading prefix tokens
                # (cls + register) if the quality seq is longer than the mask.
                if s.shape[1] > m.shape[1]:
                    s = s[:, s.shape[1] - m.shape[1]:]
                deg = (m > 0.5).float()      # degraded -> target 0
                clean = (m < 1e-6).float()   # fully clean -> target 1
                sup = deg + clean            # 1 where supervised, 0 = ignore
                target = clean               # clean=1, deg=0
                s_ = s.clamp(1e-6, 1 - 1e-6)
                bce = -(target * torch.log(s_) + (1 - target) * torch.log(1 - s_))
                denom = sup.sum().clamp_min(1.0)
                total = total + (bce * sup).sum() / denom
                z += 1
        return total / max(z, 1)

    def _mask_to_token(self, mask, grid):
        """Downsample a [B,1,H,W] degradation mask to the patch-token grid and
        flatten to [B, N] using max-pool (conservative: any degradation in the
        receptive field marks the token degraded)."""
        gh, gw = grid
        m = F.adaptive_max_pool2d(mask, (gh, gw))   # [B,1,gh,gw]
        return m.flatten(2).squeeze(1)              # [B, N]

    # ---- training: Paradigm One (internal degradation, single forward) ----
    def loss(self, inputs, data_samples):
        rgb, t = self._split(inputs)
        epoch = getattr(self, "current_epoch", 0)
        drgb, dir_, mask_rgb, mask_t = self.degrader(rgb, t, epoch=epoch)

        ml_layers, cl_layers, quality_info = self._dual_stream_forward(
            drgb, dir_, return_quality=True)

        # segmentation loss (deep supervision over all query-decode layers)
        targets = build_targets(data_samples, self.num_classes, self.ignore_index)
        losses = {}
        for li, (ml, cl) in enumerate(zip(ml_layers, cl_layers)):
            ml_up = F.interpolate(ml, size=self.img_size, mode="bilinear", align_corners=False)
            for k, v in self.criterion(ml_up, targets, cl).items():
                losses[f"l{li}.{k}"] = v

        # quality supervision (masked BCE on token grid). Only when the
        # mechanism is on -- with use_quality=False the scores are forced to 1
        # and carry no learnable signal.
        if self.use_quality and self.quality_loss_weight > 0 and quality_info:
            grid = self.network.encoder.backbone.patch_embed.grid_size
            m_rgb_tok = self._mask_to_token(mask_rgb, grid)
            m_t_tok = self._mask_to_token(mask_t, grid)
            losses["loss_quality"] = self.quality_loss_weight * self._quality_loss(
                quality_info, m_rgb_tok, m_t_tok)

        # distillation vs the frozen clean teacher (feature + output). The
        # student's _dual_stream_forward already cached its merged feature; the
        # teacher provides the clean-input targets. Gate by max(s_rgb, s_t): with
        # use_quality=False the gate is 1 everywhere (ungated, plain distill).
        if self.teacher is not None and (
                self.distill_loss_weight > 0 or self.output_distill_weight > 0):
            student_merged = self._last_merged_feat            # [B,N,C] (degraded)
            with torch.no_grad():
                t_merged, t_class_map = self.teacher.forward_distill_targets(rgb, t)
            s_rgb, s_t = quality_info[-1]
            gate_tok = torch.maximum(s_rgb, s_t)               # [B,N,1]
            if self.distill_loss_weight > 0:
                diff = (student_merged - t_merged.detach()) ** 2
                losses["loss_distill_feat"] = self.distill_loss_weight * \
                    (gate_tok * diff).mean()
            if self.output_distill_weight > 0:
                # student per-pixel class map (permutation-invariant einsum)
                student_class_map = mask_class_to_seg_logits(
                    ml_layers[-1], cl_layers[-1])
                gate_grid = self._token_gate_to_grid(
                    gate_tok, student_class_map.shape[-2:])
                losses["loss_distill_out"] = self.output_distill_weight * \
                    self._output_distill_loss(
                        student_class_map, t_class_map.detach(), gate_grid)
        return losses

    # ---- distillation helpers ----
    def _token_gate_to_grid(self, gate_tok, out_hw):
        """[B,N,1] token gate -> [B,1,h,w] at out_hw. Drops leading prefix
        tokens (cls/register) if present, reshapes to the patch grid, and
        bilinearly resizes to the class-map resolution."""
        gh, gw = self.network.encoder.backbone.patch_embed.grid_size
        g = gate_tok.squeeze(-1)                       # [B, N]
        if g.shape[1] > gh * gw:                       # drop prefix tokens
            g = g[:, g.shape[1] - gh * gw:]
        g = g.view(g.shape[0], 1, gh, gw)
        return F.interpolate(g, size=out_hw, mode="bilinear", align_corners=False)

    def _output_distill_loss(self, s_logits, t_logits, gate, eps=1e-6):
        """Permutation-invariant per-pixel KL between student and teacher class
        maps (both in [0,1] from the einsum), normalized to per-pixel
        distributions and quality-gated (relax where both modalities are weak)."""
        s = s_logits / (s_logits.sum(1, keepdim=True) + eps)
        t = t_logits / (t_logits.sum(1, keepdim=True) + eps)
        kl = (t * ((t + eps).log() - (s + eps).log())).sum(1, keepdim=True)
        return (gate * kl).mean()
