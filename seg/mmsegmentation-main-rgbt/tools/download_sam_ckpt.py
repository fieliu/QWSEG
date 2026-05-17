#!/usr/bin/env python
"""Download SAM ViT-B pretrained checkpoint."""
import os
import urllib.request

PRETRAIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'pretrain')
os.makedirs(PRETRAIN_DIR, exist_ok=True)

SAM_VIT_B_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_VIT_B_PATH = os.path.join(PRETRAIN_DIR, "sam_vit_b_01ec64.pth")

if os.path.exists(SAM_VIT_B_PATH):
    print(f"SAM ViT-B checkpoint already exists at: {SAM_VIT_B_PATH}")
else:
    print(f"Downloading SAM ViT-B checkpoint to: {SAM_VIT_B_PATH}")
    print(f"URL: {SAM_VIT_B_URL}")
    urllib.request.urlretrieve(SAM_VIT_B_URL, SAM_VIT_B_PATH)
    print("Download completed!")

print(f"Checkpoint size: {os.path.getsize(SAM_VIT_B_PATH) / 1024 / 1024:.1f} MB")
