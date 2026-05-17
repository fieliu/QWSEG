import sys
sys.path.insert(0, '/home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt')

import torch

print("Testing RGBTv1SAMNative model import and forward pass...")

from mmseg.models.segmentors.rgbt_v1_sam_native import RGBTv1SAMNative, LayerNorm2d

backbone_cfg = dict(
    type='SAMViT',
    img_size=480,
    patch_size=16,
    in_channels=3,
    embed_dims=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    out_chans=256,
    qkv_bias=True,
    use_rel_pos=True,
    rel_pos_zero_init=True,
    window_size=14,
    global_attn_indexes=(2, 5, 8, 11),
    out_indices=(3, 5, 7, 11),
    pretrained=None)

fmb_class_names = [
    'Background', 'Road', 'Sidewalk', 'Building',
    'Traffic Light', 'Traffic Sign', 'Vegetation', 'Sky',
    'Person', 'Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle', 'Pole']

model = RGBTv1SAMNative(
    backbone=backbone_cfg,
    num_classes=15,
    image_size=480,
    prompt_embed_dim=256,
    decoder_depth=2,
    decoder_num_heads=8,
    decoder_mlp_dim=2048,
    num_multimask_outputs=3,
    clip_embed_dim=768,
    clip_checkpoint=None,
    class_names=fmb_class_names,
    point_grid_size=32,
    use_lora_backbone=True,
    lora_rank=4,
    lora_alpha=4.0,
    lora_dropout=0.1,
    lora_target_modules=['qkv', 'proj'],
    use_lora_decoder=True,
    decoder_lora_rank=4,
    decoder_lora_alpha=4.0,
    decoder_lora_dropout=0.0,
    freeze_backbone_non_lora=True,
    freeze_decoder_non_lora=True,
    freeze_prompt_encoder=True,
    freeze_clip=True,
    train_patch_embed=True)

print("Model created successfully!")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")

x = torch.randn(2, 6, 480, 480)
print(f"\nInput shape: {x.shape}")

with torch.no_grad():
    image_embeddings = model.encode_image(x)
    print(f"Image embeddings shape: {image_embeddings.shape}")

    clip_text_features = model.text_encoder()
    print(f"CLIP text features shape: {clip_text_features.shape}")

    low_res_masks = model.forward_decoder(image_embeddings, clip_text_features)
    print(f"Low-res masks shape: {low_res_masks.shape}")

    seg_logits = torch.nn.functional.interpolate(
        low_res_masks, size=(480, 480), mode='bilinear', align_corners=False)
    print(f"Seg logits shape: {seg_logits.shape}")
    print(f"Expected shape: (2, 15, 480, 480)")

assert seg_logits.shape == (2, 15, 480, 480), f"Shape mismatch: {seg_logits.shape}"
print("\nAll tests passed!")
