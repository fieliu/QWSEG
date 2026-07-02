_base_ = ['./_base_/datasets/mfnet_480x640.py', './_base_/default_runtime.py']

# Override to tiny size for smoke test
train_dataloader = dict(batch_size=1, num_workers=0,
    dataset=dict(pipeline=[dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'), dict(type='Resize', scale=(160, 120), keep_ratio=False),
        dict(type='PackSegInputs')]))
val_dataloader = dict(batch_size=1, num_workers=0,
    dataset=dict(pipeline=[dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(160, 120), keep_ratio=False),
        dict(type='LoadAnnotations'), dict(type='PackSegInputs')]))
test_dataloader = val_dataloader
