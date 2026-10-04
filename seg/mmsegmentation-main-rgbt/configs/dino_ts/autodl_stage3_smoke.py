"""Sparse distillation runtime check, using a smoke teacher (not a trained one).

Pass model.teacher_ckpt=... explicitly. Tests the soft gate in training and the
real hard gather during validation. A second run may set soft_to_hard_epoch=0.
"""
_base_ = ['autodl_stage2a_smoke.py']
teacher_cfg = {{_base_.model}}
model = dict(
    type='DinoTSSparse', forward_mode='sparse_hard',
    teacher_cfg=teacher_cfg, teacher_ckpt=None, init_from_teacher=True,
    target_k=600, budget_schedule=(0.5,), budget_warmup_epochs=30,
    soft_to_hard_epoch=30, lambda_comp=1.0, lambda_rob=1.0, lambda_logit=1.0)
train_cfg = dict(max_iters=2, val_interval=2)
default_hooks = dict(checkpoint=dict(interval=2))
