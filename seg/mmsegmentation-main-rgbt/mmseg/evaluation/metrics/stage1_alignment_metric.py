"""Validation metric for the label-free DINO-TS adaptation stage."""
from collections import defaultdict
from typing import Dict, List, Sequence

from mmengine.evaluator import BaseMetric

from mmseg.registry import METRICS


@METRICS.register_module()
class Stage1AlignmentMetric(BaseMetric):
    """Average the alignment losses emitted by ``DinoTSStage1Adapt``."""

    default_prefix = 'stage1'

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            self.results.append({
                'align_loss': float(sample['align_loss']),
                'cross_patch': float(sample['cross_patch']),
                'cross_region': float(sample['cross_region']),
                'cross_relation': float(sample['cross_relation']),
            })

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        if not results:
            raise RuntimeError('Stage-1 validation produced no samples')
        values = defaultdict(list)
        for result in results:
            for key, value in result.items():
                values[key].append(value)
        return {
            key: sum(items) / len(items)
            for key, items in values.items()
        }
