# custom_imports = dict(imports=['mmseg.datasets.fmb', 'mmseg.datasets.transforms.loading'])

_base_ = [
    '../_base_/models/upernet_mae_classbalance.py',
    '../_base_/datasets/fmb_768x768.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_320k.py'
]
crop_size = (768, 768)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='./pretrain/M-SpecGene_VIT-B_seg_transform.pth',
    backbone=dict(
        type='MAE',
        img_size=(768, 768),
        patch_size=16,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        init_values=1.0,
        drop_path_rate=0.1,
        out_indices=[3, 5, 7, 11]),
    neck=dict(embed_dim=768 * 2, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[768 * 2, 768 * 2, 768 * 2, 768 * 2],
        num_classes=15,
        channels=768),
    auxiliary_head=dict(in_channels=768 * 2, num_classes=15),
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(341, 341)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65),
    constructor='LayerDecayOptimizerConstructor')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=320000,
        by_epoch=False)
]

fp16 = dict(loss_scale='dynamic')

train_dataloader = dict(batch_size=1)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
