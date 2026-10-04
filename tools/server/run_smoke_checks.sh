#!/usr/bin/env bash
set -euo pipefail
ROOT=${QWSEG_ROOT:-/root/autodl-tmp/code/QWSEG}
export PATH="${QWSEG_ENV:-/root/autodl-tmp/envs/qwseg}/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/seg/mmsegmentation-main-rgbt:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
cd "$ROOT/seg/mmsegmentation-main-rgbt"
latest_checkpoint() {
  python -c 'from pathlib import Path; import sys; print(max(Path(sys.argv[1]).rglob("iter_*.pth"), key=lambda p: p.stat().st_mtime))' "$1"
}
check_checkpoint() {
  python "$ROOT/tools/server/check_smoke_checkpoint.py" "$(latest_checkpoint "$ROOT/work_dirs/$1")" --steps "$2"
}
python tools/train.py configs/dino_ts/autodl_stage1_llvip_smoke.py --work-dir "$ROOT/work_dirs/smoke_stage1"
check_checkpoint smoke_stage1 4
python tools/train.py configs/dino_ts/autodl_stage2a_smoke.py --work-dir "$ROOT/work_dirs/smoke_stage2a"
check_checkpoint smoke_stage2a 4
python tools/train.py configs/dino_ts/autodl_stage2a_robust_smoke.py --work-dir "$ROOT/work_dirs/smoke_stage2a_robust"
check_checkpoint smoke_stage2a_robust 2
# Tiny smoke teacher only: this checkpoint has no scientific performance claim.
CKPT=$(latest_checkpoint "$ROOT/work_dirs/smoke_stage2a_robust")
python tools/train.py configs/dino_ts/autodl_stage2b_smoke.py --work-dir "$ROOT/work_dirs/smoke_stage2b" --cfg-options load_from="$CKPT"
check_checkpoint smoke_stage2b 2
python tools/train.py configs/dino_ts/autodl_stage3_smoke.py --work-dir "$ROOT/work_dirs/smoke_stage3_soft" --cfg-options model.teacher_ckpt="$CKPT"
python "$ROOT/tools/server/check_smoke_checkpoint.py" "$(latest_checkpoint "$ROOT/work_dirs/smoke_stage3_soft")" --steps 2 --teacher "$CKPT" --require-router-update
python tools/train.py configs/dino_ts/autodl_stage3_smoke.py --work-dir "$ROOT/work_dirs/smoke_stage3_hard" --cfg-options model.teacher_ckpt="$CKPT" model.soft_to_hard_epoch=0
python "$ROOT/tools/server/check_smoke_checkpoint.py" "$(latest_checkpoint "$ROOT/work_dirs/smoke_stage3_hard")" --steps 2 --teacher "$CKPT"
printf 'ALL_SMOKES_COMPLETED\n'
