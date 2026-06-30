"""Multimodal (RGB-T) EoMT with shared ViT + cross-attention fusion.

Deliverable 2 (baseline, no quality):
- RGB and Thermal share ONE DINOv3 ViT (Siamese / shared weights).
- The two streams run in parallel through the early blocks; at several fusion
  points (evenly distributed BEFORE the query-decode stage) they exchange
  information via cross-attention, then keep propagating.
- Just before EoMT inserts its learnable queries (block L - num_blocks), the
  two streams are merged into a single N-token sequence (simple mean here),
  after which the original EoMT query-decode logic is reused unchanged.

Input is the project's 6-channel RGB-T tensor (RGB = first 3, T = last 3),
already normalized by the mmseg data_preprocessor.
"""
import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .eomt_segmentor import EoMTSegmentor
from .eomt_fusion_blocks import CrossAttnFusion


@MODELS.register_module()
class EoMTRGBTFusion(EoMTSegmentor):
    def __init__(self, *args, num_fusion_points=3, fusion_heads=8,
                 fusion_points=None,
                 use_contrast=False, contrast_weight=0.1,
                 contrast_layer=None, contrast_samples=15,
                 contrast_tau=0.1, patch_size=16, **kwargs):
        super().__init__(*args, patch_size=patch_size, **kwargs)
        self.patch_size = patch_size
        self.fusion_heads = fusion_heads
        net = self.network
        num_layers = len(net.encoder.backbone.blocks)
        self.decode_start = num_layers - net.num_blocks  # query inserted here

        # Fusion points: either explicitly specified or evenly placed in the
        # dual-stream stage [0, decode_start). Explicit specification allows
        # non-uniform placement (e.g. denser in deeper layers for better
        # cross-modal semantic alignment, following the hierarchical-ViT
        # principle that deep layers need more fusion). Points are clamped to
        # [1, decode_start-1] and sorted/deduped.
        if fusion_points is not None:
            pts = sorted({min(max(int(p), 1), self.decode_start - 1)
                          for p in fusion_points})
            self.fusion_points = pts
        elif num_fusion_points > 0 and self.decode_start > 0:
            step = max(1, self.decode_start // (num_fusion_points + 1))
            pts = [min((j + 1) * step, self.decode_start - 1)
                   for j in range(num_fusion_points)]
            self.fusion_points = sorted(set(pts))
        else:
            self.fusion_points = []

        dim = net.encoder.backbone.embed_dim
        # one cross-attn fusion module per fusion point (RGB and T share it both
        # directions by calling it twice with swapped args).
        self.fusions = nn.ModuleList(
            [CrossAttnFusion(dim, num_heads=fusion_heads) for _ in self.fusion_points]
        )

        # ---- 跨模态对比学习 (可控开关) ----
        # 在指定层的融合前双流特征上做 CLIP 风格双向 InfoNCE,
        # 让 T 向 RGB 对齐, 缓解 RGB 缺失时性能下降.
        # contrast_layer=None 时默认用最后一个融合点 (深层, 语义最强).
        self.use_contrast = use_contrast
        self.contrast_weight = contrast_weight
        self.contrast_samples = contrast_samples
        self.contrast_tau = contrast_tau
        if contrast_layer is None:
            # 默认: 最后一个融合点 (深层, 语义最强, 融合前双流独立)
            self.contrast_layer = self.fusion_points[-1] if self.fusion_points else (self.decode_start - 1)
        else:
            self.contrast_layer = min(int(contrast_layer), self.decode_start - 1)
        # token grid (h_tok, w_tok), 用于把像素级标签下采样到 token 分辨率
        h_tok = self.img_size[0] // self.patch_size
        w_tok = self.img_size[1] // self.patch_size
        self._token_grid = (h_tok, w_tok)

    # -- split 6ch input into RGB and T (both 3ch) --
    def _split(self, inputs):
        return inputs[:, :3], inputs[:, 3:]

    def _run_block(self, block, net, x, attn_mask, rope):
        attn = block.attn if hasattr(block, "attn") else block.attention
        attn_out = net._attn(attn, block.norm1(x), attn_mask, rope=rope)
        if hasattr(block, "ls1"):
            attn_out = block.ls1(attn_out)
        elif hasattr(block, "layer_scale1"):
            attn_out = block.layer_scale1(attn_out)
        x = x + block.drop_path(attn_out)

        mlp_out = block.mlp(block.norm2(x))
        if hasattr(block, "ls2"):
            mlp_out = block.ls2(mlp_out)
        elif hasattr(block, "layer_scale2"):
            mlp_out = block.layer_scale2(mlp_out)
        x = x + block.drop_path(mlp_out)
        return x

    def _merge(self, z_rgb, z_t, **kwargs):
        """Baseline merge: simple mean of the two streams."""
        return 0.5 * (z_rgb + z_t)

    # -- dual-stream forward: shared ViT + cross-attn fusion, then EoMT decode --
    def _dual_stream_forward(self, rgb, t, return_quality=False):
        net = self.network
        backbone = net.encoder.backbone
        num_layers = len(backbone.blocks)

        # patch embed + pos embed (shared weights, applied to each modality)
        def embed(x):
            r = backbone.rope_embeddings(x) if hasattr(backbone, "rope_embeddings") else None
            x = backbone.patch_embed(x)
            if hasattr(backbone, "_pos_embed"):
                x = backbone._pos_embed(x)
            return x, r

        z_rgb, rope = embed(rgb)
        z_t, _ = embed(t)

        fp_to_idx = {p: i for i, p in enumerate(self.fusion_points)}
        quality_info = []  # reserved for the quality subclass hook
        # 保存对比层融合前的双流特征 (供 loss 计算对比损失)
        self._contrast_feat = None

        # ---- dual-stream stage: blocks [0, decode_start) ----
        for i in range(self.decode_start):
            block = backbone.blocks[i]
            # 先保存对比特征 (融合前双流独立, 语义已成熟),
            # 必须在 _fuse 之前, 否则保存的是融合后两流已互相污染的特征,
            # 对比损失信号会大幅减弱甚至失效.
            if self.use_contrast and i == self.contrast_layer:
                self._contrast_feat = (z_rgb, z_t)
            if i in fp_to_idx:
                z_rgb, z_t = self._fuse(fp_to_idx[i], z_rgb, z_t, quality_info)
            z_rgb = self._run_block(block, net, z_rgb, None, rope)
            z_t = self._run_block(block, net, z_t, None, rope)

        # ---- merge into a single token sequence ----
        z = self._merge(z_rgb, z_t, quality_info=quality_info)
        self._last_merged_feat = z  # [B,N,C], exposed as a distillation target

        # ---- EoMT query-decode stage: blocks [decode_start, num_layers) ----
        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []
        x = z
        for i in range(self.decode_start, num_layers):
            block = backbone.blocks[i]
            if i == self.decode_start:
                x = torch.cat(
                    (net.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )
            if net.masked_attn_enabled:
                ml, cl = net._predict(backbone.norm(x))
                mask_logits_per_layer.append(ml)
                class_logits_per_layer.append(cl)
                attn_mask = net._attn_mask(x, ml, i)
            x = self._run_block(block, net, x, attn_mask, rope)

        ml, cl = net._predict(backbone.norm(x))
        mask_logits_per_layer.append(ml)
        class_logits_per_layer.append(cl)

        if return_quality:
            return mask_logits_per_layer, class_logits_per_layer, quality_info
        return mask_logits_per_layer, class_logits_per_layer

    def _fuse(self, fidx, z_rgb, z_t, quality_info):
        """Baseline: symmetric cross-attention, no quality mask."""
        fusion = self.fusions[fidx]
        new_rgb = fusion(z_rgb, z_t, keep_mask=None)
        new_t = fusion(z_t, z_rgb, keep_mask=None)
        return new_rgb, new_t

    # -- override the single-modal forward used by loss/predict/_forward --
    def _network_forward(self, x):
        # not used: RGB-T model overrides the callers below to pass both modalities
        raise RuntimeError("EoMTRGBTFusion uses _dual_stream_forward")

    def loss(self, inputs, data_samples):
        import torch.nn.functional as F
        from .eomt_utils import build_targets
        rgb, t = self._split(inputs)
        ml_layers, cl_layers = self._dual_stream_forward(rgb, t)
        targets = build_targets(data_samples, self.num_classes, self.ignore_index)
        losses = {}
        for li, (ml, cl) in enumerate(zip(ml_layers, cl_layers)):
            ml_up = F.interpolate(ml, size=self.img_size, mode="bilinear", align_corners=False)
            for k, v in self.criterion(ml_up, targets, cl).items():
                losses[f"l{li}.{k}"] = v

        # ---- 跨模态对比损失 (可选) ----
        if self.use_contrast and self._contrast_feat is not None:
            from .cross_modal_contrast import (
                cross_modal_contrast_loss,
                _downsample_labels_to_token,
            )
            # 从 data_samples 取像素级标签 [B, H, W]
            labels_hw = torch.stack(
                [ds.gt_sem_seg.data.squeeze(0) for ds in data_samples], dim=0
            )  # [B, H, W]
            z_rgb_c, z_t_c = self._contrast_feat  # [B, N, C]
            labels_tok = _downsample_labels_to_token(labels_hw, self._token_grid)
            loss_c = cross_modal_contrast_loss(
                z_rgb_c, z_t_c, labels_tok,
                samples_per_class=self.contrast_samples,
                tau=self.contrast_tau,
                ignore_index=self.ignore_index,
            )
            losses["loss_contrast"] = loss_c * self.contrast_weight
        return losses

    def encode_decode(self, inputs, batch_img_metas):
        from .eomt_utils import mask_class_to_seg_logits, resize_seg_logits
        rgb, t = self._split(inputs)
        ml_layers, cl_layers = self._dual_stream_forward(rgb, t)
        seg_logits = mask_class_to_seg_logits(ml_layers[-1], cl_layers[-1])
        return resize_seg_logits(seg_logits, inputs.shape[2:])

    def _forward(self, inputs, data_samples=None):
        from .eomt_utils import mask_class_to_seg_logits
        rgb, t = self._split(inputs)
        ml_layers, cl_layers = self._dual_stream_forward(rgb, t)
        return mask_class_to_seg_logits(ml_layers[-1], cl_layers[-1])

    def predict_with_missing(self, inputs, data_samples=None,
                             mask_rgb=False, mask_t=False):
        """Predict with one modality zeroed out (whole-modality missing).

        Zeroing matches the MissingModalityEvalHook protocol: RGB = channels
        0:3, T = channels 3:6. Reuses predict() so postprocess (pad-crop,
        flip, resize-to-ori) is identical to the clean path. Inherited by
        EoMTRGBTQuality unchanged.
        """
        inputs = inputs.clone()
        if mask_rgb:
            inputs[:, :3] = 0
        if mask_t:
            inputs[:, 3:] = 0
        return self.predict(inputs, data_samples)

    @torch.no_grad()
    def forward_distill_targets(self, rgb, t):
        """Clean-input distillation targets for the student (E1/E2).

        Runs one clean dual-stream forward and returns:
          - merged_feat [B, N, C]: the post-fusion merged token sequence
            (feature-distillation target, matches the student's
            self._last_merged_feat),
          - mask_t [B, Q, h, w]: teacher per-query mask logits (raw, pre-sigmoid)
          - cls_t  [B, Q, num_classes+1]: teacher per-query class logits (raw)
        The query-level logits are used for DETRDistill-style query-matched
        distillation (Hungarian-match student queries to teacher predictions as
        pseudo-GT), which avoids the per-pixel einsum-map normalization that
        degenerates in background regions.
        """
        ml_layers, cl_layers = self._dual_stream_forward(rgb, t)
        return self._last_merged_feat, ml_layers[-1], cl_layers[-1]

