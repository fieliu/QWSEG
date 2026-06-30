# DINOv3 + ViT-Adapter + Mask2Former: T-only baseline on MFNet.
# Following official ViT-Adapter config pattern.
# Uses the last 3 channels (Thermal) of the 6-channel RGB-T input.
_base_ = ['./adapter_m2f_rgb_mfnet.py']

model = dict(
    backbone=dict(use_thermal=True),  # T only
)
