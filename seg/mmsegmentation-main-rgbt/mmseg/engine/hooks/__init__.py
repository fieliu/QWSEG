try:
    from .train_vis_hook import TrainVisHook
except ImportError:
    TrainVisHook = None
from .visualization_hook import SegVisualizationHook
from .val_degradation_hook import ValDegradationHook

if TrainVisHook is not None:
    __all__ = ['SegVisualizationHook', 'TrainVisHook', 'ValDegradationHook']
else:
    __all__ = ['SegVisualizationHook', 'ValDegradationHook']
