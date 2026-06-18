"""Teacher model: clean dual-branch shared-Swin Mask2Former with concat+reduce
fusion. Trained on CLEAN RGB-T. Its per-stage fused features and final logits
serve as distillation targets for the student.

Structurally identical to SwinMulV6Mask2Former except the fusion is
concat-then-1x1conv (channel reduce) instead of addition, per the user's spec.
"""
import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .swinmul_v6_mask2former import SwinMulV6Mask2Former


@MODELS.register_module()
class QualityDistillTeacher(SwinMulV6Mask2Former):
    def __init__(self, *args, fusion_dims=(96, 192, 384, 768), **kwargs):
        super().__init__(*args, **kwargs)
        # concat (2C) -> 1x1 conv -> C, one per stage
        self.fuse_convs = nn.ModuleList(
            [nn.Conv2d(c * 2, c, 1) for c in fusion_dims]
        )

    def _fuse_stage(self, i, x_rgb, x_t):
        return self.fuse_convs[i](torch.cat([x_rgb, x_t], dim=1))

    def extract_feat(self, inputs: torch.Tensor):
        B = inputs.shape[0] // 2
        x_rgbt = self.backbone(inputs)
        x_rgb_list = [feat[:B] for feat in x_rgbt]
        x_t_list = [feat[B:] for feat in x_rgbt]

        self._last_rgb_feats = x_rgb_list
        self._last_t_feats = x_t_list

        fused_feats = []
        for i in range(len(x_rgb_list)):
            fused_feats.append(self._fuse_stage(i, x_rgb_list[i], x_t_list[i]))

        self._last_fused_feats = fused_feats  # exposed for distillation
        if self.with_neck:
            fused_feats = self.neck(fused_feats)
        return fused_feats

    @torch.no_grad()
    def extract_feat_vis(self, inputs):
        """Visualization hook entry. inputs: 6ch RGB-T OR batch-doubled cat.
        Teacher is the clean target: RGB feats | T feats | fused feats. No
        quality, no degradation."""
        if inputs.shape[1] == 6:
            rgb, t = inputs[:, :3], inputs[:, 3:]
        else:
            B = inputs.shape[0] // 2
            rgb, t = inputs[:B], inputs[B:]
        self.extract_feat(torch.cat([rgb, t], dim=0))
        return dict(
            zc_rgb=list(self._last_rgb_feats),
            zc_t=list(self._last_t_feats),
            final_fused=list(self._last_fused_feats),
            clean_rgb_img=rgb, clean_t_img=t)

        """inputs: 6-channel RGB-T. Returns list of pre-neck fused stage feats."""
        input_rgb, input_ir = inputs[:, :3], inputs[:, 3:]
        rgbt = torch.cat([input_rgb, input_ir], dim=0)
        self.extract_feat(rgbt)
        return self._last_fused_feats
