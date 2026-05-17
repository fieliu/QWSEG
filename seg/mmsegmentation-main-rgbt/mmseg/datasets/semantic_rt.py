from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class SemanticRTDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('Background', 'Car_Stop', 'Bike', 'Bicyclist',
                 'Motorcycle', 'Motorcyclist', 'Car', 'Tricycle',
                 'Traffic_Light', 'Box', 'Pole', 'Curve', 'Person'),
        palette=[[0, 0, 0], [72, 61, 39], [0, 0, 255], [148, 0, 211],
                 [128, 128, 0], [64, 64, 128], [0, 139, 139],
                 [131, 139, 139], [192, 64, 0], [126, 192, 238],
                 [244, 164, 96], [211, 211, 211], [205, 155, 155]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
