"""Utility to load DINOv3 backbone for use outside the EoMT segmentor.

Reuses the same ViT wrapper from eomt_core/vit.py for consistent weight loading.
"""
from typing import Optional
import torch.nn as nn


def load_dinov3_backbone(
    backbone_name: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    backbone_ckpt: Optional[str] = None,
    img_size: tuple = (480, 640),
    patch_size: int = 16,
    local_files_only: bool = True,
) -> nn.Module:
    """Load and return a DINOv3 ViT backbone with timm-compatible interface.

    Returns the backbone with .blocks, .patch_embed, .embed_dim, etc.
    """
    import importlib
    # Import eomt_core.vit directly without triggering segmentors/__init__.py
    vit_mod = importlib.import_module('mmseg.models.segmentors.eomt_core.vit')
    ViT = vit_mod.ViT

    encoder = ViT(
        img_size=img_size,
        patch_size=patch_size,
        backbone_name=backbone_name,
        ckpt_path=backbone_ckpt,
        local_files_only=local_files_only,
    )
    return encoder.backbone
