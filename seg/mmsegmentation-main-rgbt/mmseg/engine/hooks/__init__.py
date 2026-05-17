try:
    from .train_vis_hook import TrainVisHook
except ImportError:
    TrainVisHook = None
from .visualization_hook import SegVisualizationHook

if TrainVisHook is not None:
    __all__ = ['SegVisualizationHook', 'TrainVisHook']
else:
    __all__ = ['SegVisualizationHook']
