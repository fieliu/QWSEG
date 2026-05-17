from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MFNetDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('unlabeled', 'car', 'person', 'bike', 'curve',
                 'car_stop', 'guardrail', 'color_cone', 'bump'),
        palette=[[0, 0, 0], [0, 0, 142], [0, 60, 100], [0, 0, 230],
                 [119, 11, 32], [255, 0, 0], [0, 139, 139],
                 [255, 165, 150], [192, 64, 0]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
