import os
import urllib.request

pretrain_dir = '/home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt/pretrain'
os.makedirs(pretrain_dir, exist_ok=True)

sam_url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
sam_path = os.path.join(pretrain_dir, 'sam_vit_b_01ec64.pth')

clip_url = 'https://download.openmmlab.com/mmsegmentation/v0.5/san/clip_vit-base-patch16-224_3rdparty-d08f8887.pth'
clip_path = os.path.join(pretrain_dir, 'clip_vit_base_patch16_224.pth')

if not os.path.exists(sam_path):
    print(f'Downloading SAM ViT-B checkpoint to {sam_path}...')
    urllib.request.urlretrieve(sam_url, sam_path)
    print(f'SAM checkpoint downloaded: {os.path.getsize(sam_path) / 1e9:.2f} GB')
else:
    print(f'SAM checkpoint exists: {os.path.getsize(sam_path) / 1e9:.2f} GB')

if not os.path.exists(clip_path):
    print(f'Downloading CLIP ViT-B/16 checkpoint to {clip_path}...')
    urllib.request.urlretrieve(clip_url, clip_path)
    print(f'CLIP checkpoint downloaded: {os.path.getsize(clip_path) / 1e6:.2f} MB')
else:
    print(f'CLIP checkpoint exists: {os.path.getsize(clip_path) / 1e6:.2f} MB')

print('All checkpoints ready!')
