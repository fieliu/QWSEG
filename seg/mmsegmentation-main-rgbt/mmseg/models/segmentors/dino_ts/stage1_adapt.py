"""Stage 1: Modality Adaptation (doc section 6).

Goal: make the Thermal PatchEmbed + shallow Thermal Adapters produce tokens the
shared DINO attention/FFN can process, aligned to the common semantics of the
frozen RGB DINO teacher — WITHOUT requiring Thermal to fully copy RGB.

Frozen (doc 6.3): RGB PatchEmbed, shared DINO backbone, teacher AlignmentProjector,
Fusion / Router, decode head.
Trained: Thermal PatchEmbed + modality embedding, shallow Thermal Adapters,
Student AlignmentProjector.

Loss (doc 6.4):
    L_stage1 = lambda_patch  * L_cross_patch
             + lambda_region * L_cross_region
Teacher backbone + projector are fixed, so no L_var is needed (doc 6.4).
No unlabeled semantic pseudo-labels are produced.

This stage has no segmentation output; it is a pure feature-alignment pretext
run on paired (well-registered) RGB-T. It reuses DinoTSBase only for the shared
backbone; loss()/predict() are overridden.
"""
import torch

from mmseg.registry import MODELS
from .base_dino_ts import DinoTSBase
from . import losses as L


@MODELS.register_module()
class DinoTSStage1Adapt(DinoTSBase):
    def __init__(self, *args,
                 lambda_patch: float = 1.0,
                 lambda_region: float = 1.0,
                 region: int = 2,
                 **kwargs):
        kwargs.setdefault('forward_mode', 'adapt')
        super().__init__(*args, **kwargs)
        self.lambda_patch = lambda_patch
        self.lambda_region = lambda_region
        self.region = region
        self._freeze_for_stage1()

    def _freeze_for_stage1(self):
        bb = self.backbone
        # freeze shared DINO backbone
        if getattr(bb, 'backbone', None) is not None:
            for p in bb.backbone.parameters():
                p.requires_grad = False
        # freeze RGB patch embed + RGB modality embed + RGB adapters
        for p in bb.patch_embed['rgb'].parameters():
            p.requires_grad = False
        bb.modality_embed['rgb'].requires_grad = False
        for ad in bb.adapters['rgb']:
            for p in ad.parameters():
                p.requires_grad = False
        # freeze fusion, router, teacher projector, decode head (unused here)
        for mod in (bb.fusion, bb.router, bb.teacher_projector,
                    self.neck, self.decode_head):
            for p in mod.parameters():
                p.requires_grad = False
        bb.anchor_pos_embed.requires_grad = False

    def loss(self, inputs, data_samples):
        rgb, thermal = self._split(inputs)
        out = self.backbone(rgb, thermal, mode='adapt')
        r_tok, t_tok = out['rgb_tokens'], out['thermal_tokens']
        H, W = out['grid']

        # teacher = frozen RGB projector on RGB tokens (stop-grad target);
        # student = trainable projector on Thermal tokens
        with torch.no_grad():
            z_rgb = self.backbone.teacher_projector(r_tok, 'rgb')
        z_thr = self.backbone.student_projector(t_tok, 'thermal')

        losses = dict()
        if self.lambda_patch > 0:
            losses['loss_cross_patch'] = self.lambda_patch * L.cross_patch_loss(
                z_thr, z_rgb)
        if self.lambda_region > 0:
            losses['loss_cross_region'] = self.lambda_region * L.cross_region_loss(
                z_thr, z_rgb, (H, W), region=self.region)
        return losses

    def predict(self, inputs, data_samples=None):
        # Stage 1 produces no segmentation; return the (unmodified) samples so a
        # ValLoop won't crash. Real Stage-1 exit criteria (doc 6.5) use a linear
        # probe, evaluated by a dedicated tool, not this predict path.
        return data_samples if data_samples is not None else []
