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


@DATASETS.register_module()
class FMBDataset13(FMBDataset):
    """FMB with Bicycle removed: 13 foreground classes.

    Identical to :class:`FMBDataset` except that ``Bicycle`` is gone from the
    class list, so the evaluator reports a 13-class mean.  Pair this with
    ``DropLabels(labels=(12,))`` in the pipeline -- after ``reduce_zero_label``
    the raw label 13 (Bicycle) becomes index 12, and those pixels must be set
    to the ignore value or the model would be asked to predict a class the
    metric no longer knows about.

    Why drop it: across the 1,220 training images Bicycle occupies 9,115 pixels
    in 12 images, and in the 280-image validation split it has exactly zero
    pixels.  Its IoU is undefined there, and a spurious prediction would give a
    non-empty union with an empty intersection and pull the mean down.
    """
    METAINFO = dict(
        classes=('Road', 'Sidewalk', 'Building',
                 'Traffic Light', 'Traffic Sign', 'Vegetation', 'Sky',
                 'Person', 'Car', 'Truck', 'Bus', 'Motorcycle',
                 'Pole'),
        palette=[[0, 0, 142], [0, 60, 100], [0, 0, 230],
                 [119, 11, 32], [255, 0, 0], [0, 139, 139],
                 [255, 165, 150], [192, 64, 0], [211, 211, 211],
                 [100, 33, 128], [117, 79, 86], [153, 153, 153],
                 [250, 170, 30]])
