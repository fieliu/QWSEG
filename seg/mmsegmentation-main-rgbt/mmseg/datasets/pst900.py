from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class PST900Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('Background', 'Fire_Extinguisher', 'Backpack',
                 'Hand_Drill', 'Survivor'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0],
                 [128, 128, 0], [0, 0, 128]])

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
