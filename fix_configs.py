import os, glob

base = '/home/lh/code/QWSEG/seg/mmsegmentation-main-rgbt/configs'
patterns = [
    'mae/mae-base_upernet_rgbt_v1_baseline_8xb2-amp-200e_*.py',
    'mae/mae-base_upernet_rgbt_v2_disentangle_8xb2-amp-200e_*.py',
    'mae/mae-base_upernet_rgbt_v3_degradation_8xb2-amp-300e_*.py',
    'mae/mae-base_upernet_rgbt_v4_quality_pruning_8xb2-amp-300e_*.py',
    'mae/mae-base_upernet_rgbt_v5_quality_joint_8xb2-amp-300e_*.py',
    'sam/sam-base_upernet_rgbt_v1_sam_baseline_8xb2-amp-200e_*.py',
]
files = []
for p in patterns:
    files.extend(glob.glob(os.path.join(base, p)))

changed = 0
for f in files:
    with open(f, 'r') as fh:
        content = fh.read()
    new_content = content.replace('val_interval=20', 'val_interval=10')
    new_content = new_content.replace(
        "checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=20)",
        "checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=10)"
    )
    if new_content != content:
        with open(f, 'w') as fh:
            fh.write(new_content)
        changed += 1
        print(f'Updated: {os.path.basename(f)}')

print(f'Total files changed: {changed}')
