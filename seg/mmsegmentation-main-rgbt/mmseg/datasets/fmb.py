from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FMBDataset(BaseSegDataset):
    # 14 foreground classes (Background excluded, following the common FMB
    # evaluation protocol). With reduce_zero_label=True, raw label 0
    # (Background) is mapped to ignore (255) and labels 1..14 -> 0..13.
    METAINFO = dict(
        classes=('Road', 'Sidewalk', 'Building',
                 'Traffic Light', 'Traffic Sign', 'Vegetation', 'Sky',
                 'Person', 'Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle',
                 'Pole'),
        palette=[[0, 0, 142], [0, 60, 100], [0, 0, 230],
                 [119, 11, 32], [255, 0, 0], [0, 139, 139],
                 [255, 165, 150], [192, 64, 0], [211, 211, 211],
                 [100, 33, 128], [117, 79, 86], [153, 153, 153],
                 [190, 122, 222], [250, 170, 30]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
