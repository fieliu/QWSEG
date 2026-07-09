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


def _load_meta_dinov3_pth(ckpt_path):
    """Build an HF DINOv3ViTModel and load a RAW Meta DINOv3 .pth into it.

    The Meta checkpoint is a plain state_dict with the original facebookresearch
    naming (cls_token / storage_tokens / blocks.N.attn.qkv / ls1.gamma / ...).
    HF uses different names and a SPLIT q/k/v projection (k has no bias). We
    construct the config from the checkpoint itself (size-agnostic) and remap.
    """
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("model", sd.get("state_dict", sd))

    hidden = sd["cls_token"].shape[-1]
    n_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    inter = sd["blocks.0.mlp.fc1.weight"].shape[0]
    n_reg = sd["storage_tokens"].shape[1] if "storage_tokens" in sd else 0
    patch = sd["patch_embed.proj.weight"].shape[-1]
    cfg = DINOv3ViTConfig(
        patch_size=patch, hidden_size=hidden, intermediate_size=inter,
        num_hidden_layers=n_layers, num_attention_heads=hidden // 64,
        num_register_tokens=n_reg, use_gated_mlp=False,
        rope_theta=100.0)
    model = DINOv3ViTModel(cfg)

    new = {}
    new["embeddings.cls_token"] = sd["cls_token"]
    if "storage_tokens" in sd:
        new["embeddings.register_tokens"] = sd["storage_tokens"]
    if "mask_token" in sd:
        new["embeddings.mask_token"] = sd["mask_token"].reshape(1, 1, -1)
    new["embeddings.patch_embeddings.weight"] = sd["patch_embed.proj.weight"]
    new["embeddings.patch_embeddings.bias"] = sd["patch_embed.proj.bias"]
    if "rope_embed.periods" in sd:
        new["rope_embeddings.periods"] = sd["rope_embed.periods"]
    new["norm.weight"] = sd["norm.weight"]
    new["norm.bias"] = sd["norm.bias"]

    for i in range(n_layers):
        s, d = f"blocks.{i}.", f"model.layer.{i}."
        new[d + "norm1.weight"] = sd[s + "norm1.weight"]
        new[d + "norm1.bias"] = sd[s + "norm1.bias"]
        new[d + "norm2.weight"] = sd[s + "norm2.weight"]
        new[d + "norm2.bias"] = sd[s + "norm2.bias"]
        # split fused qkv -> q/k/v (k has no bias in HF; drop its bias slice)
        qw, kw, vw = sd[s + "attn.qkv.weight"].chunk(3, dim=0)
        new[d + "attention.q_proj.weight"] = qw
        new[d + "attention.k_proj.weight"] = kw
        new[d + "attention.v_proj.weight"] = vw
        if s + "attn.qkv.bias" in sd:
            qb, kb, vb = sd[s + "attn.qkv.bias"].chunk(3, dim=0)
            new[d + "attention.q_proj.bias"] = qb
            new[d + "attention.v_proj.bias"] = vb
        new[d + "attention.o_proj.weight"] = sd[s + "attn.proj.weight"]
        new[d + "attention.o_proj.bias"] = sd[s + "attn.proj.bias"]
        new[d + "layer_scale1.lambda1"] = sd[s + "ls1.gamma"]
        new[d + "layer_scale2.lambda1"] = sd[s + "ls2.gamma"]
        new[d + "mlp.up_proj.weight"] = sd[s + "mlp.fc1.weight"]
        new[d + "mlp.up_proj.bias"] = sd[s + "mlp.fc1.bias"]
        new[d + "mlp.down_proj.weight"] = sd[s + "mlp.fc2.weight"]
        new[d + "mlp.down_proj.bias"] = sd[s + "mlp.fc2.bias"]

    missing, unexpected = model.load_state_dict(new, strict=False)
    real_missing = [k for k in missing if "rope" not in k and "k_proj.bias" not in k]
    print(f"[DINOv3 .pth load] mapped={len(new)} "
          f"missing={len(real_missing)} unexpected={len(unexpected)}")
    if real_missing:
        print("  missing sample:", real_missing[:5])
    if unexpected:
        print("  unexpected sample:", unexpected[:5])
    return model


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
            if src is not None and str(src).endswith(".pth"):
                # Raw Meta DINOv3 .pth (state_dict only): build the HF arch in
                # code (no download) and load with Meta->HF key remapping.
                backbone = _load_meta_dinov3_pth(src)
            else:
                backbone = AutoModel.from_pretrained(
                    src, local_files_only=local_files_only)
            self.backbone = self.transformers_to_timm(backbone, img_size)
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
        # DINOv3ViTModel structure varies by transformers version:
        #   Old: backbone.model.layer (DINOv3ViTEncoder nested under .model)
        #   New: backbone.layer (DINOv3ViTEncoder flattened into top-level)
        # DINOv2: backbone.layers (timm style)
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

        # Auto-detect layer location
        if hasattr(backbone, 'model') and hasattr(backbone.model, 'layer'):
            # Old DINOv3: layers under backbone.model.layer
            backbone.blocks = backbone.model.layer
            del backbone.model
        elif hasattr(backbone, 'layer'):
            # New DINOv3: layers directly on backbone
            backbone.blocks = backbone.layer
            del backbone.layer
        elif hasattr(backbone, 'layers'):
            # DINOv2 / timm style
            backbone.blocks = backbone.layers

        # Clean up embeddings reference (we use patch_embed now)
        del backbone.embeddings
        # Remove mask_token if present (not needed for inference)
        if hasattr(backbone, 'mask_token'):
            del backbone.mask_token

        return backbone
