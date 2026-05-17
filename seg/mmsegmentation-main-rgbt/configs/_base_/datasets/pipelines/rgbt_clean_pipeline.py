"""
数据增强管道配置 - 干净数据 (V1/V2使用)

无退化增强，仅使用标准几何和颜色增强。
"""

train_pipeline_clean = [
    dict(type='LoadRGBTImageFromFile',
         ir_replace_src='FMB_ALL/FMB',
         ir_replace_dst='FMB_ALL/FMB_T'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2560, 768),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(768, 768), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
