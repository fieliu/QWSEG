# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import Optional
import torch
import torch.nn as nn

try:
    import timm
except Exception:  # pragma: no cover
    timm = None
try:
    from transformers import AutoModel
except Exception:  # pragma: no cover
    AutoModel = None


class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size=16,
        backbone_name="vit_large_patch14_reg4_dinov2",
        ckpt_path: Optional[str] = None,
        local_files_only: bool = True,
    ):
        super().__init__()

        # HuggingFace-style name ("facebook/dinov3-...") OR a local directory
        # path containing a saved HF model -> load via transformers AutoModel.
        # timm-style name ("vit_base_patch14_reg4_dinov2") -> load via timm.
        is_hf = ("/" in backbone_name) or (ckpt_path is not None and "dinov3" in str(backbone_name).lower())

        if is_hf:
            assert AutoModel is not None, "transformers is required for DINOv3 backbones"
            # ckpt_path (a local dir) takes precedence so we can run fully offline.
            src = ckpt_path if ckpt_path is not None else backbone_name
            self.backbone = self.transformers_to_timm(
                AutoModel.from_pretrained(
                    src,
                    local_files_only=local_files_only,
                ),
                img_size,
            )
        else:
            assert timm is not None, "timm is required for DINOv2 backbones"
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=ckpt_path is None,
                img_size=img_size,
                patch_size=patch_size,
                num_classes=0,
            )
            if ckpt_path is not None:
                state = torch.load(ckpt_path, map_location="cpu")
                state = state.get("model", state)
                self.backbone.load_state_dict(state, strict=False)

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def transformers_to_timm(self, backbone, img_size: tuple[int, int]):
        backbone.patch_embed = backbone.embeddings
        backbone.patch_embed.patch_size = (
            backbone.embeddings.config.patch_size,
            backbone.embeddings.config.patch_size,
        )
        backbone.patch_embed.grid_size = (
            img_size[0] // backbone.embeddings.config.patch_size,
            img_size[1] // backbone.embeddings.config.patch_size,
        )

        backbone.embed_dim = backbone.embeddings.config.hidden_size
        backbone.num_prefix_tokens = backbone.patch_embed.config.num_register_tokens + 1
        backbone.blocks = backbone.layer

        del (
            backbone.patch_embed.mask_token,
            backbone.embeddings,
            backbone.layer,
        )

        return backbone
