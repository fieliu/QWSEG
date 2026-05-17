"""
数据增强管道配置 - 中度退化 (V3/V4标准版)

退化由模型内部处理，数据管道不施加退化。
"""

train_pipeline_medium_degradation = [
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
