"""Two real-data optimizer steps through clean/degraded/missing views."""
_base_ = ['autodl_stage2a_smoke.py']
model = dict(lambda_deg=1.0, lambda_missing=1.0)
train_cfg = dict(max_iters=2, val_interval=2)
default_hooks = dict(checkpoint=dict(interval=2))
