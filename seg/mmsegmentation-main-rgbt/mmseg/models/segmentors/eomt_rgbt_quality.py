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
from .eomt_fusion_blocks import TokenQualityPredictor, CrossAttnFusion
from .degradation import DegradationGenerator
from .eomt_utils import build_targets, mask_class_to_seg_logits, resize_seg_logits
from .eomt_quality_attn import (
    wrap_backbone_attention,
    soft_keep_mask, complementary_fix_tokens)
from .distill_utils import prepare_ssl_outputs


@MODELS.register_module()
class EoMTRGBTQuality(EoMTRGBTFusion):
    def __init__(self, *args,
                 quality_loss_weight=1.0,
                 fuse_tau=0.5,
                 quality_tau=0.5,
                 mask_temperature=0.1,
                 rank_margin=0.20,
                 deg_ceiling_weight=0.1,
                 deg_ceiling=0.2,
                 use_quality=True,
                 use_self_attn_bias=True,
                 use_compensation=True,
                 teacher_cfg=None,
                 teacher_ckpt=None,
                 init_from_teacher=True,
                 distill_loss_weight=1.0,
                 output_distill_weight=1.0,
                 distill_temperature=4.0,
                 clean_floor_weight=0.1,
                 clean_floor=0.9,
                 freeze_backbone=False,
                 freeze_neck_head=False,
                 degradation=None,
                 # Dormant params (accepted for config backward-compat, unused)
                 suppress_value=-20.0,
                 bias_alpha=4.0,
                 bias_gamma=3.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        dim = self.network.encoder.backbone.embed_dim

        # Quality is evaluated at SEGMENT boundaries: after embedding (segment 0)
        # and after each fusion point. One predictor per segment boundary. The
        # sigmoid score s is converted to a SOFT keep-mask D = sigmoid((s-tau)/T)
        # (fully differentiable, gradient D(1-D)/T focuses on boundary tokens,
        # inspired by DynamicViT's Gumbel-Softmax) + soft complementary fix;
        # D drives attention/fusion KEY-side suppression (source-blocking).
        self.num_segments = len(self.fusion_points) + 1
        self.quality_predictors = nn.ModuleList(
            [TokenQualityPredictor(dim) for _ in range(self.num_segments)]
        )
        # Dedicated cross-attn fusion for the final merge. Unlike the per-segment
        # fusions (which run mid-stream at fusion_points), this runs ONCE right
        # before the two streams are merged into a single sequence for the EoMT
        # decoder. It aligns RGB/T tokens via cross-attention so that the
        # subsequent quality-weighted mean operates on semantically aligned
        # features (not raw unaligned tokens). Shares fusion_heads with the
        # per-segment fusions. Reuses the keep-mask key-gate so low-quality keys
        # of the OTHER modality are proportionally suppressed during alignment.
        self.merge_fusion = CrossAttnFusion(dim, num_heads=self.fusion_heads)
        self.quality_loss_weight = quality_loss_weight
        self.fuse_tau = fuse_tau
        self.quality_tau = quality_tau
        self.mask_temperature = mask_temperature
        # rank_margin: fixed score separation in rank loss. With clean_floor=0.9
        # and deg_ceiling=0.2, the usable score range is ~0.7. margin=0.20 gives
        # clear separation between light/heavy without exceeding the range.
        self.rank_margin = rank_margin
        # suppress_value is DORMANT: the multiplicative key-gate does not use
        # an additive bias, so this value is unused. Kept as a config option
        # for backward-compat (old configs still pass it) and for ablation
        # (additive-bias variant reactives it via quality_mask_to_bias).
        self.suppress_value = suppress_value
        # deg_ceiling anchors the LOWER end of quality scores: heavy-degraded
        # tokens are pushed below deg_ceiling (default 0.2) so the soft mask
        # D=sigmoid((s-0.5)/0.1) fires strongly (s=0.2 -> D=0.001). Combined
        # with clean_floor=0.9 (upper anchor) + rank_loss (monotonicity), the
        # score range is pinned to [0.2, 0.9] -> D range [0.001, 0.999].
        self.deg_ceiling_weight = deg_ceiling_weight
        self.deg_ceiling = deg_ceiling
        # bias_alpha / bias_gamma / use_compensation are accepted for config
        # backward-compat but DORMANT: the convex soft bias and the cross-modal
        # compensation module were replaced by the soft-mask + cross-attn repair
        # design (low-quality tokens are key-suppressed and repaired by attending
        # to the other modality's kept tokens, not zeroed / soft-blended).
        self.bias_alpha = bias_alpha
        self.bias_gamma = bias_gamma
        # master switch: use_quality=False (E0b/E1) forces s=1 -> D=1 everywhere,
        # so the multiplicative key-gate (attn * 1 = attn), complementary_fix
        # (no both-prune) and the merge (softmax([1,1]/tau)=mean) all collapse
        # to the EoMTRGBTFusion baseline. use_self_attn_bias is slaved to it so
        # attention wrapping can't re-enable the mechanism when the master is off.
        self.use_quality = use_quality
        self.use_self_attn_bias = use_self_attn_bias and use_quality
        self.use_compensation = use_compensation and use_quality  # dormant

        # wrap each backbone block's attention so we can inject a multiplicative
        # quality key-gate into the modality-internal self-attention. Done AFTER
        # the teacher warm-start below would change key names, so the warm-start
        # must run first -> defer wrapping to the end of __init__.

        self.distill_loss_weight = distill_loss_weight
        self.output_distill_weight = output_distill_weight
        self.distill_temperature = distill_temperature
        self.clean_floor_weight = clean_floor_weight
        self.clean_floor = clean_floor

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

        # frozen-backbone experiment: keep the ORIGINAL pretrained backbone
        # frozen, train only the increments (quality / compensation / fusion)
        # and the task head. Validates the "detachable robustness increment"
        # claim. QualityAttnWrapper has no params of its own, so freezing all of
        # backbone.parameters() freezes exactly the original ViT weights; the
        # quality modules live under separate attributes and stay trainable.
        if freeze_backbone:
            n_frozen = 0
            for p in self.network.encoder.backbone.parameters():
                p.requires_grad = False
                n_frozen += 1
            print_log(
                f'EoMTRGBTQuality: freeze_backbone=True -> froze {n_frozen} '
                f'backbone param tensors; quality/fusion/decode-head trainable.',
                logger='current')
        # PURE-PLUGIN (Fp): on top of freeze_backbone, also freeze fusion +
        # decode head (q / class_head / mask_head / upscale), leaving ONLY the
        # quality predictors + compensation + attn-bias as trainable increment.
        # Isolates the quality mechanism's value with the host model fully frozen
        # (true LoRA-style plugin). attn bias is param-free; clean input -> s~=1,
        # bias~=0 -> equals the frozen host, so clean perf does not drop.
        if freeze_neck_head:
            n_fnh = 0
            mods = [self.fusions, self.network.q, self.network.class_head,
                    self.network.mask_head, self.network.upscale]
            for m in mods:
                for p in m.parameters():
                    p.requires_grad = False
                    n_fnh += 1
            print_log(
                f'EoMTRGBTQuality: freeze_neck_head=True -> additionally froze '
                f'{n_fnh} fusion+decode-head param tensors; ONLY quality '
                f'predictor/compensation trainable (pure-plugin Fp).',
                logger='current')
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

    # ---- run one block on one stream, optionally injecting a self-attn mask ----
    def _run_block_q(self, block, net, x, rope, keep_mask):
        if self.use_self_attn_bias and keep_mask is not None:
            attn_mod = block.attn if hasattr(block, 'attn') else block.attention
            attn_mod.set_keep_mask(keep_mask)
        return self._run_block(block, net, x, None, rope)

    @property
    def _num_prefix_tokens(self):
        """Number of leading prefix tokens (cls + register) that must NEVER be
        quality-suppressed. DINOv3 has 1 cls + N register tokens; Swin has 0.
        These tokens carry global information and suppressing them as keys
        would starve all patch tokens of global context."""
        backbone = self.network.encoder.backbone
        n_reg = getattr(getattr(backbone, 'config', None), 'num_register_tokens', 0)
        n_cls = 1 if hasattr(getattr(backbone, 'embeddings', None), 'cls_token') else 0
        return n_cls + n_reg

    def _protect_prefix_mask(self, D):
        """Set D=1 for the leading prefix tokens so they are never suppressed.
        D: [B, N, 1] (soft keep-mask) -> [B, N, 1] with prefix rows = 1.
        Returns a cloned tensor so the original D (used elsewhere) is untouched."""
        n = self._num_prefix_tokens
        if n > 0:
            D = D.clone()
            D[:, :n] = 1.0
        return D

    def _mask_2d(self, D):
        """Squeeze [B,N,1] -> [B,N] for the attention wrapper / fusion."""
        return D.squeeze(-1)

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
        (D_rgb, s_rgb), (D_t, s_t) = self._eval_quality(seg, z_rgb, z_t)
        all_quality.append((s_rgb, s_t))   # store soft scores for vis + losses
        m_rgb = self._mask_2d(self._protect_prefix_mask(D_rgb))
        m_t = self._mask_2d(self._protect_prefix_mask(D_t))

        for i in range(self.decode_start):
            block = backbone.blocks[i]
            if i in fp_to_seg:
                # fuse (cross-attn: each modality perceives the other's KEPT
                # tokens; low-quality keys suppressed), then re-evaluate quality
                # for the NEW segment on the (repaired) fused tokens.
                z_rgb, z_t = self._fuse_q(z_rgb, z_t, D_rgb, D_t)
                seg = fp_to_seg[i]
                (D_rgb, s_rgb), (D_t, s_t) = self._eval_quality(seg, z_rgb, z_t)
                all_quality.append((s_rgb, s_t))
                # record this fusion stage: pre-merge streams + their merge
                # (vis only -- skip during training to avoid extra merge cost)
                if not self.training:
                    vis_rgb.append(z_rgb)
                    vis_t.append(z_t)
                    vis_fused.append(self._merge_q(z_rgb, z_t, s_rgb, s_t))
                m_rgb = self._mask_2d(self._protect_prefix_mask(D_rgb))
                m_t = self._mask_2d(self._protect_prefix_mask(D_t))
            # each stream suppresses ITS OWN low-quality tokens as keys:
            z_rgb = self._run_block_q(block, net, z_rgb, rope, m_rgb)
            z_t = self._run_block_q(block, net, z_t, rope, m_t)

        # quality-weighted merge of the two (repaired) streams
        z = self._merge_q(z_rgb, z_t, s_rgb, s_t)

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
        # Returns ((D_rgb, s_rgb), (D_t, s_t)):
        #   s = soft sigmoid score [B,N,1] (for supervision + vis, stored in
        #      all_quality; rank/clean_floor/deg_ceiling/gate all consume s).
        #   D = complementary-fixed SOFT keep-mask [B,N,1] (for attention /
        #      fusion key-suppression). D=sigmoid((s-tau)/T) in (0,1).
        # use_quality=False -> D=s=1 everywhere -> key-bias=0, complementary_fix
        # no-op, merge=mean -> exactly the EoMTRGBTFusion baseline.
        if not self.use_quality:
            ones = z_rgb.new_ones(z_rgb.shape[0], z_rgb.shape[1], 1)
            return (ones, ones), (ones, ones)
        qp = self.quality_predictors[seg]
        s_rgb = qp(z_rgb, z_t)               # [B,N,1]
        s_t = qp(z_t, z_rgb)
        D_rgb = soft_keep_mask(s_rgb, self.quality_tau, self.mask_temperature)
        D_t = soft_keep_mask(s_t, self.quality_tau, self.mask_temperature)
        D_rgb, D_t = complementary_fix_tokens(D_rgb, D_t, s_rgb, s_t)
        return (D_rgb, s_rgb), (D_t, s_t)

    def _fuse_q(self, z_rgb, z_t, D_rgb, D_t):
        """Cross-attn fusion: each modality perceives the OTHER's high-quality
        tokens. Low-quality tokens of the key modality are key-suppressed
        (invisible as keys) via the soft keep-mask D -> RGB attends only to T's
        kept tokens and vice versa; each modality's own low-quality tokens act
        as queries and are repaired by aggregating the other's good tokens
        (residual add in CrossAttnFusion). With multiplicative key-gating,
        suppression is truly proportional (D=0.5 -> key weight x0.5,
        D=0.01 -> x0.01), unlike an additive bias which softmax amplifies
        exponentially."""
        m_t = self._mask_2d(self._protect_prefix_mask(D_t))
        m_rgb = self._mask_2d(self._protect_prefix_mask(D_rgb))
        fusion = self.fusions[self._fuse_call]
        new_rgb = fusion(z_rgb, z_t, keep_mask=m_t)   # RGB attends to T; low-q T keys suppressed
        new_t = fusion(z_t, z_rgb, keep_mask=m_rgb)
        self._fuse_call += 1
        return new_rgb, new_t

    def _merge_q(self, z_rgb, z_t, s_rgb, s_t):
        # Cross-attn alignment + quality-weighted merge of the two (repaired)
        # streams. The per-segment fusions already align features mid-stream,
        # but the LAST segment runs blocks after the last fusion_point without
        # any cross-modal interaction, so the two streams drift apart before the
        # merge. This dedicated merge_fusion re-aligns them right before the
        # weighted mean: each modality attends to the OTHER's KEPT tokens
        # (low-quality keys suppressed via the keep-mask), then the aligned
        # features are quality-weighted into a single sequence for the decoder.
        # s=1 (use_quality=False) -> D=1 -> keep_mask=1 -> cross-attn is a plain
        # bidirectional attention (no suppression); softmax([1,1]/tau)=0.5/0.5
        # -> mean of aligned features. With use_quality=True the merge is a
        # quality-weighted mean of cross-aligned features.
        if not self.use_quality:
            # D=1 everywhere: still align via cross-attn, then plain mean.
            # This is NOT identical to the EoMTRGBTFusion baseline (which does a
            # raw mean with no alignment), but the alignment is a strict
            # improvement and the baseline path is recovered by setting
            # use_quality=False AND not using merge_fusion (handled by the
            # parent class _merge). Here we always align for robustness.
            z_rgb_a = self.merge_fusion(z_rgb, z_t, keep_mask=None)
            z_t_a = self.merge_fusion(z_t, z_rgb, keep_mask=None)
            return 0.5 * (z_rgb_a + z_t_a)
        D_rgb = soft_keep_mask(s_rgb, self.quality_tau, self.mask_temperature)
        D_t = soft_keep_mask(s_t, self.quality_tau, self.mask_temperature)
        m_rgb = self._mask_2d(self._protect_prefix_mask(D_rgb))
        m_t = self._mask_2d(self._protect_prefix_mask(D_t))
        # Bidirectional cross-attn alignment with quality key-gate:
        # RGB attends to T's kept tokens (T's low-q keys suppressed),
        # T attends to RGB's kept tokens. Residual add preserves each stream.
        z_rgb_a = self.merge_fusion(z_rgb, z_t, keep_mask=m_t)
        z_t_a = self.merge_fusion(z_t, z_rgb, keep_mask=m_rgb)
        # Quality-weighted mean of the ALIGNED features.
        w = torch.softmax(torch.cat([s_rgb, s_t], dim=-1) / self.fuse_tau, dim=-1)
        return w[..., 0:1] * z_rgb_a + w[..., 1:2] * z_t_a


    # ---- quality supervision (masked BCE) ----
    def _rank_loss(self, quality_info, lvl_light_rgb_tok, lvl_light_t_tok,
                   lvl_heavy_rgb_tok, lvl_heavy_t_tok, B,
                   margin=0.20, rank_mask=None, level_gap=None):
        """Rank quality loss using LEVEL masks (0=clean, 1-5=degraded).

        For each token in the degraded region, push s_light > s_heavy by a
        FIXED margin (not scaled by level gap):
            L = mean max(0, margin - (s_light - s_heavy))
        The level gap is already reflected in the IMAGE (higher level = stronger
        degradation), so the score separation naturally grows with gap. The rank
        loss only enforces MONOTONICITY (s_light > s_heavy + margin), not a
        proportional separation. Scaling margin by gap would demand impossible
        separations (e.g. gap=4 -> margin=0.40, but s in [0,1]).

        margin=0.20: with clean_floor=0.9 and deg_ceiling=0.2, the usable score
        range is ~0.7. 0.20 gives clear separation without exceeding the range.

        The level masks encode the ACTUAL degradation level per token:
          - 0 = clean (outside region, or lvl_lo=0 inside)
          - 1-5 = degradation level (inside region)
        Tokens where lvl_heavy == lvl_light (equal levels) have no rank signal
        (identical images) -> excluded by reg mask (lvl_h > 0 AND lvl_l < lvl_h).

        Args:
            quality_info: list of (s_rgb, s_t), each [2B,N,1] (light=0:B,
                heavy=B:2B).
            lvl_light_rgb_tok, lvl_light_t_tok: [B,N] long, level labels for
                the LIGHT version (from level_light masks).
            lvl_heavy_rgb_tok, lvl_heavy_t_tok: [B,N] long, level labels for
                the HEAVY version.
            B: batch size.
            margin: fixed score separation (default 0.10).
            rank_mask: [B] float, 1 if valid pair (lvl_hi > lvl_lo), 0 skip.
            level_gap: [B] float, unused (kept for call-site compat).
        """
        total = z = 0.0
        for (s_rgb, s_t) in quality_info:
            for s, lvl_l, lvl_h in (
                    (s_rgb, lvl_light_rgb_tok, lvl_heavy_rgb_tok),
                    (s_t, lvl_light_t_tok, lvl_heavy_t_tok)):
                s = s.squeeze(-1)                       # [2B, N]
                if s.shape[1] > lvl_l.shape[1]:
                    s = s[:, s.shape[1] - lvl_l.shape[1]:]
                s_light, s_heavy = s[:B], s[B:]         # [B, N] each
                gap = s_light - s_heavy                 # want > margin
                hinge = (margin - gap).clamp_min(0.0)   # [B, N]
                # Region mask: only where heavy is degraded AND light < heavy
                # (valid rank pair). Equal levels -> identical images -> skip.
                # 0=clean, 1-5=degraded: reg where lvl_h > 0 AND lvl_l < lvl_h
                reg = ((lvl_h > 0) & (lvl_l < lvl_h)).float()  # [B, N]
                # Apply sample-level rank_mask (skip equal-level pairs)
                if rank_mask is not None:
                    hinge = hinge * rank_mask.view(B, 1)
                    reg = reg * rank_mask.view(B, 1)
                denom = reg.sum().clamp_min(1.0)
                total = total + (hinge * reg).sum() / denom
                z += 1
        return total / max(z, 1)

    def _clean_floor_loss(self, quality_info, lvl_light_tok, lvl_heavy_tok, B):
        """Anti-collapse soft floor on CLEAN tokens (level == 0). Both light
        and heavy versions (full 2B) should keep quality >= clean_floor where
        the level is 0 (clean). Soft lower bound relu(floor - s): only
        penalizes scores BELOW the floor, leaving [floor, 1] free.

        lvl_light_tok, lvl_heavy_tok: [B,N] long, level labels for light/heavy.
        Tiled to 2B: light levels [0:B], heavy levels [B:2B]."""
        floor = self.clean_floor
        total = z = 0.0
        for (s_rgb, s_t) in quality_info:
            for s, lvl_l, lvl_h in (
                    (s_rgb, lvl_light_tok[0], lvl_heavy_tok[0]),
                    (s_t, lvl_light_tok[1], lvl_heavy_tok[1])):
                s = s.squeeze(-1)                       # [2B, N]
                if s.shape[1] > lvl_l.shape[1]:
                    s = s[:, s.shape[1] - lvl_l.shape[1]:]
                # Clean tokens: level == 0. Tile light+heavy levels.
                lvl_all = torch.cat([lvl_l, lvl_h], dim=0)  # [2B, N]
                clean = (lvl_all == 0).float()              # [2B, N]
                below = (floor - s).clamp_min(0.0)          # penalize s<floor
                denom = clean.sum().clamp_min(1.0)
                total = total + (below * clean).sum() / denom
                z += 1
        return total / max(z, 1)

    def _deg_ceiling_loss(self, quality_info, lvl_light_tok, lvl_heavy_tok, B):
        """Lower-end anchor: on the HEAVY version's DEGRADED tokens (level >=
        1), push s_heavy BELOW deg_ceiling. The ceiling scales with level:
        higher level -> lower ceiling (stronger suppression target).
            ceiling(lvl) = deg_ceiling * (6 - lvl) / 5  # L1->base, L5->0.2*base
        This gives a CONTINUOUS anchor: L5 (worst) -> s<=0.04, L1 (mild) ->
        s<=0.20. Combined with clean_floor=0.9 (L0), the score range is pinned
        to [0.04, 0.9], monotonically decreasing with level.

        lvl_heavy_tok: [B,N] long, level labels for heavy version."""
        base_ceiling = self.deg_ceiling
        total = z = 0.0
        for (s_rgb, s_t) in quality_info:
            for s, lvl_h in (
                    (s_rgb, lvl_heavy_tok[0]),
                    (s_t, lvl_heavy_tok[1])):
                s = s.squeeze(-1)                       # [2B, N]
                if s.shape[1] > lvl_h.shape[1]:
                    s = s[:, s.shape[1] - lvl_h.shape[1]:]
                s_heavy = s[B:]                         # [B, N] heavy version
                # Level-dependent ceiling: L1->base, L5->0.2*base
                # ceiling = base * (6 - lvl) / 5  (lvl in 1-5)
                lvl_f = lvl_h.float()
                ceiling = base_ceiling * (6.0 - lvl_f) / 5.0  # [B, N]
                above = (s_heavy - ceiling).clamp_min(0.0)     # 0 where s<=ceiling
                reg = (lvl_h > 0).float()                      # degraded region
                denom = reg.sum().clamp_min(1.0)
                total = total + (above * reg).sum() / denom
                z += 1
        return total / max(z, 1)

    def _quality_loss(self, quality_info, mask_rgb_tok, mask_t_tok):
        """DEPRECATED (not called). Old binary BCE quality supervision
        (degraded->0, clean->1). Replaced by _rank_loss (paired light/heavy
        ranking) because binary supervision collapsed the quality score to a
        2-valued detector. Kept for a potential BCE-vs-rank ablation."""
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
        """Downsample a [B,1,H,W] mask to the patch-token grid and flatten to
        [B, N]. Uses max-pool for binary masks (conservative: any degradation
        in the receptive field marks the token). For level masks (long), uses
        max-pool too (conservative: take the HIGHEST level in the receptive
        field, so severe degradation is never missed)."""
        gh, gw = grid
        m = F.adaptive_max_pool2d(mask.float(), (gh, gw))  # [B,1,gh,gw]
        return m.flatten(2).squeeze(1).long()              # [B, N]

    # ---- training: Paradigm One (internal degradation, single forward) ----
    def loss(self, inputs, data_samples):
        # E2 (use_quality=True): PAIRED multi-level degradation for quality
        # ranking. E0b/E1 (use_quality=False): single-level path (unchanged).
        if self.use_quality:
            return self._loss_paired(inputs, data_samples)
        return self._loss_single(inputs, data_samples)

    def _seg_losses(self, ml_layers, cl_layers, targets, prefix=""):
        out = {}
        for li, (ml, cl) in enumerate(zip(ml_layers, cl_layers)):
            ml_up = F.interpolate(ml, size=self.img_size, mode="bilinear",
                                  align_corners=False)
            for k, v in self.criterion(ml_up, targets, cl).items():
                out[f"{prefix}l{li}.{k}"] = v
        return out

    # ---- E0b/E1 path: single-level degradation, no quality ranking ----
    def _loss_single(self, inputs, data_samples):
        rgb, t = self._split(inputs)
        epoch = getattr(self, "current_epoch", 0)
        drgb, dir_, mask_rgb, mask_t = self.degrader(rgb, t, epoch=epoch)

        ml_layers, cl_layers, quality_info = self._dual_stream_forward(
            drgb, dir_, return_quality=True)

        targets = build_targets(data_samples, self.num_classes, self.ignore_index)
        losses = self._seg_losses(ml_layers, cl_layers, targets)
        self._add_distill_losses(losses, rgb, t, ml_layers, cl_layers,
                                 quality_info)
        return losses

    # ---- E2 path: paired light/heavy degradation + rank-only quality loss ----
    def _loss_paired(self, inputs, data_samples):
        rgb, t = self._split(inputs)
        epoch = getattr(self, "current_epoch", 0)
        mean = self.data_preprocessor.mean.flatten()
        std = self.data_preprocessor.std.flatten()
        (l_rgb, l_ir, h_rgb, h_ir,
         lvl_light_rgb, lvl_light_ir,
         lvl_heavy_rgb, lvl_heavy_ir,
         rank_mask, level_gap) = \
            self.degrader.make_paired(rgb, t, mean, std, epoch=epoch)
        B = rgb.shape[0]

        # batch-cat light(0:B) + heavy(B:2B), one forward
        cat_rgb = torch.cat([l_rgb, h_rgb], dim=0)
        cat_ir = torch.cat([l_ir, h_ir], dim=0)
        ml_layers, cl_layers, quality_info = self._dual_stream_forward(
            cat_rgb, cat_ir, return_quality=True)

        # segmentation loss on BOTH versions (targets duplicated light+heavy)
        targets = build_targets(data_samples, self.num_classes, self.ignore_index)
        targets2 = targets + targets
        losses = self._seg_losses(ml_layers, cl_layers, targets2)

        # quality supervision using LEVEL masks (L1-L5)
        if self.quality_loss_weight > 0 and quality_info:
            grid = self.network.encoder.backbone.patch_embed.grid_size
            lvl_light_rgb_tok = self._mask_to_token(lvl_light_rgb, grid)  # [B,N]
            lvl_light_t_tok = self._mask_to_token(lvl_light_ir, grid)
            lvl_heavy_rgb_tok = self._mask_to_token(lvl_heavy_rgb, grid)
            lvl_heavy_t_tok = self._mask_to_token(lvl_heavy_ir, grid)

            losses["loss_quality"] = self.quality_loss_weight * self._rank_loss(
                quality_info,
                lvl_light_rgb_tok, lvl_light_t_tok,
                lvl_heavy_rgb_tok, lvl_heavy_t_tok,
                B, margin=self.rank_margin,
                rank_mask=rank_mask, level_gap=level_gap)
            # clean-region soft floor: where level==L1 (clean), quality should
            # not drop below clean_floor (anti-collapse). Soft lower bound
            # relu(floor - s), NOT a hard push to 1 -> clean-but-low-quality
            # regions (e.g. night RGB shadows) are not forced to lie.
            if self.clean_floor_weight > 0:
                losses["loss_clean_floor"] = self.clean_floor_weight * \
                    self._clean_floor_loss(
                        quality_info,
                        (lvl_light_rgb_tok, lvl_light_t_tok),
                        (lvl_heavy_rgb_tok, lvl_heavy_t_tok), B)
            # degraded-side ceiling with level-dependent target (DISABLED by
            # default): L5->s<=0.04, L2->s<=0.16. Kept for ablation.
            if self.deg_ceiling_weight > 0:
                losses["loss_deg_ceiling"] = self.deg_ceiling_weight * \
                    self._deg_ceiling_loss(
                        quality_info,
                        (lvl_light_rgb_tok, lvl_light_t_tok),
                        (lvl_heavy_rgb_tok, lvl_heavy_t_tok), B)

        # distillation: align the LIGHT version (closer to the clean teacher) so
        # the heavy version's lower quality is not forcibly pulled clean (which
        # would wash out the ranking signal). Slice [:B] = light everywhere.
        light_ml = [m[:B] for m in ml_layers]
        light_cl = [c[:B] for c in cl_layers]
        light_q = [(sr[:B], st[:B]) for (sr, st) in quality_info]
        # _last_merged_feat is [2B,N,C]; keep light half for feature distill
        full_merged = self._last_merged_feat
        self._last_merged_feat = full_merged[:B]
        self._add_distill_losses(losses, rgb, t, light_ml, light_cl, light_q)
        self._last_merged_feat = full_merged  # restore (vis may read it)
        return losses

    def _add_distill_losses(self, losses, rgb, t, ml_layers, cl_layers,
                            quality_info):
        """Feature + output distillation vs the frozen clean teacher. Shared by
        the single and paired paths. ml/cl/quality_info here are the LIGHT (or
        only) version, sliced to batch B; teacher consumes the clean RGB-T."""
        if self.teacher is None or (
                self.distill_loss_weight <= 0 and self.output_distill_weight <= 0):
            return
        student_merged = self._last_merged_feat            # [B,N,C]
        with torch.no_grad():
            t_merged, t_mask, t_cls = self.teacher.forward_distill_targets(rgb, t)
        s_rgb, s_t = quality_info[-1]
        gate_tok = torch.maximum(s_rgb, s_t)               # [B,N,1]
        if self.distill_loss_weight > 0:
            # L2-normalize per-token feature vectors before MSE so the loss is a
            # direction (cosine-like) match, not dominated by large-magnitude
            # channels / overall scale differences (more standard than raw MSE).
            sf = F.normalize(student_merged, dim=-1)
            tf = F.normalize(t_merged.detach(), dim=-1)
            diff = (sf - tf) ** 2
            losses["loss_distill_feat"] = self.distill_loss_weight * \
                (gate_tok * diff).mean()
        if self.output_distill_weight > 0:
            # Standard Mask2Former distillation (GuidedDistillation): turn the
            # teacher's confident-foreground queries into pseudo-GT, then feed
            # them to the model's OWN criterion (Hungarian match + class CE +
            # mask BCE + dice, point-sampled). Reuses the exact standard loss —
            # no hand-rolled matching / full-image BCE (which collapsed fg mIoU).
            s_cls, s_mask = cl_layers[-1], ml_layers[-1]   # [B,Q,C+1], [B,Q,h,w]
            pseudo = prepare_ssl_outputs(t_cls.detach(), t_mask.detach())
            d = self.criterion(s_mask, pseudo, s_cls)
            losses["loss_distill_out"] = self.output_distill_weight * sum(
                v for v in d.values())

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
