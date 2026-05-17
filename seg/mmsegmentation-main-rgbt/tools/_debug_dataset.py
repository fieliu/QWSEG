import sys; sys.path.insert(0, '.')
from mmengine.registry import init_default_scope; init_default_scope('mmseg')
from mmseg.registry import DATASETS
import mmseg.datasets; import mmseg.datasets.mfnet; import mmseg.datasets.transforms.loading
from mmseg.datasets.transforms import Resize
from mmseg.datasets.transforms.loading import LoadRGBTImageFrom4Channel
from mmengine.config import Config

cfg = Config.fromfile('configs/segformer/mitmul_v10_quality_embed_mit-b2-b0_1xb2-40K_mfnet-480x640.py')
ds_cfg = cfg.train_dataloader.dataset
dataset_cls = DATASETS.get(ds_cfg.type)
q_load = LoadRGBTImageFrom4Channel(to_float32=True)
q_pipeline = [q_load, Resize(scale=(640, 480), keep_ratio=False)]
q_ds_cfg = {k: v for k, v in ds_cfg.items() if k not in ('type', 'pipeline')}
quality_dataset = dataset_cls(pipeline=q_pipeline, **q_ds_cfg)
sample0 = quality_dataset[0]
print('type:', type(sample0).__name__)
if isinstance(sample0, dict):
    for k, v in sample0.items():
        if hasattr(v, 'shape'):
            print(f'  {k}: {type(v).__name__} shape={v.shape}')
        else:
            print(f'  {k}: {type(v).__name__}')
else:
    print('NOT a dict')
    if hasattr(sample0, 'keys'):
        for k in sample0.keys():
            v = sample0[k]
            if hasattr(v, 'shape'):
                print(f'  {k}: {type(v).__name__} shape={v.shape}')
            else:
                print(f'  {k}: {type(v).__name__}')
    else:
        print(dir(sample0))
