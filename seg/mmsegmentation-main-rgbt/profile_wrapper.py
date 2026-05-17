#!/usr/bin/env python3
"""Profile V9 training speed without modifying source code.

Usage:
    python profile_wrapper.py
    The first 3 training iterations will be profiled, then results printed.
    A chrome trace file `v9_trace.json` will also be saved.
"""

import sys
import os

# Point to the mmseg project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from torch.profiler import profile, ProfilerActivity, schedule

PROFILE_WARMUP = 1    # warmup iter (skip)
PROFILE_ACTIVE = 3    # number of iters to profile
PROFILE_WAIT = 0      # skip iters between warmup and active

_iter_counter = 0
_prof = None


def _patched_loss(self, inputs, data_samples):
    global _iter_counter, _prof

    if _iter_counter < PROFILE_WARMUP:
        _iter_counter += 1
        return _original_loss(self, inputs, data_samples)

    if _iter_counter == PROFILE_WARMUP:
        _prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        )
        _prof.__enter__()

    result = _original_loss(self, inputs, data_samples)

    if _prof is not None:
        _prof.step()
        _iter_counter += 1

        if _iter_counter >= PROFILE_WARMUP + PROFILE_ACTIVE:
            _prof.__exit__(None, None, None)

            # Print top CUDA kernels
            print("\n" + "=" * 80)
            print("TOP 20 CUDA KERNELS BY TIME")
            print("=" * 80)
            print(
                _prof.key_averages().table(
                    sort_by="cuda_time_total", row_limit=20
                )
            )

            # Print top CPU ops
            print("\n" + "=" * 80)
            print("TOP 10 CPU OPS BY TIME")
            print("=" * 80)
            print(
                _prof.key_averages().table(
                    sort_by="cpu_time_total", row_limit=10
                )
            )

            # Export chrome trace
            trace_path = os.path.join(PROJECT_ROOT, "v9_trace.json")
            _prof.export_chrome_trace(trace_path)
            print(f"\nChrome trace saved to: {trace_path}")
            print("Open chrome://tracing in Chrome and load this file.")
            _prof = None

    return result


# Monkey-patch before importing the rest of mmseg
from mmseg.models.segmentors.mitmul_ablation import MiTMulABV9

_original_loss = MiTMulABV9.loss
MiTMulABV9.loss = _patched_loss

# Now run training
if __name__ == "__main__":
    from tools.train import main

    main()
