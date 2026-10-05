#!/usr/bin/env bash
# Formal DINO-TS stage launcher for one or multiple GPUs.
#
# Examples:
#   GPUS=4 bash tools/server/train_dino_ts.sh stage1
#   GPUS=4 bash tools/server/train_dino_ts.sh stage2a
#   GPUS=4 RESUME=1 bash tools/server/train_dino_ts.sh stage2a
#   GPUS=4 INIT_CKPT=/path/to/best.pth bash tools/server/train_dino_ts.sh stage2b
set -euo pipefail

ROOT=${QWSEG_ROOT:-/root/autodl-tmp/code/QWSEG}
ENV_DIR=${QWSEG_ENV:-/root/autodl-tmp/envs/qwseg}
DATA_ROOT=${QWSEG_DATA_ROOT:-/root/autodl-tmp/data}
PRETRAIN_ROOT=${QWSEG_PRETRAIN_ROOT:-/root/autodl-tmp/pretrain}
STAGE=${1:?Usage: train_dino_ts.sh stage1|stage2a|stage2b|stage3}
GPUS=${GPUS:-1}
RESUME=${RESUME:-0}
WORK_ROOT=${WORK_ROOT:-$ROOT/work_dirs/dino_ts}
WORK_DIR=${WORK_DIR:-$WORK_ROOT/$STAGE}
MIN_FREE_GB=${MIN_FREE_GB:-10}
DRY_RUN=${DRY_RUN:-0}

export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/seg/mmsegmentation-main-rgbt:${PYTHONPATH:-}"
export DINOV3_CHECKPOINT=${DINOV3_CHECKPOINT:-$PRETRAIN_ROOT/dinov3-vitb16}
export MFNET_ROOT=${MFNET_ROOT:-$DATA_ROOT/MFNet}
export LLVIP_ROOT=${LLVIP_ROOT:-$DATA_ROOT/LLVIP}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

cd "$ROOT/seg/mmsegmentation-main-rgbt"

latest_checkpoint() {
  local directory=$1
  local pattern=$2
  python -c 'from pathlib import Path; import sys; xs=list(Path(sys.argv[1]).glob(sys.argv[2])); sys.exit("No matching checkpoint under " + sys.argv[1]) if not xs else print(max(xs, key=lambda p: p.stat().st_mtime))' "$directory" "$pattern"
}

case "$STAGE" in
  stage1)
    CONFIG=configs/dino_ts/stage1_adapt_llvip.py
    ;;
  stage2a)
    CONFIG=configs/dino_ts/stage2a_dense_mfnet.py
    ;;
  stage2b)
    CONFIG=configs/dino_ts/stage2b_ema_mfnet.py
    ;;
  stage3)
    CONFIG=configs/dino_ts/stage3_sparse_mfnet.py
    ;;
  *)
    echo "Unknown stage: $STAGE (expected stage1, stage2a, stage2b, or stage3)" >&2
    exit 2
    ;;
esac

ARGS=("$CONFIG" --work-dir "$WORK_DIR")

if [[ "$RESUME" == "1" ]]; then
  ARGS+=(--resume)
else
  case "$STAGE" in
    stage2a)
      INIT_CKPT=${INIT_CKPT:-$(latest_checkpoint "$WORK_ROOT/stage1/weight" 'best_stage1_align_loss*.pth')}
      echo "Initialization checkpoint: $INIT_CKPT"
      ARGS+=(--cfg-options "load_from=$INIT_CKPT")
      ;;
    stage2b)
      INIT_CKPT=${INIT_CKPT:-$(latest_checkpoint "$WORK_ROOT/stage2a/weight" 'best_mIoU*.pth')}
      echo "Initialization checkpoint: $INIT_CKPT"
      ARGS+=(--cfg-options "load_from=$INIT_CKPT")
      ;;
  esac
fi

# Stage 3 needs the accepted EMA teacher both for a fresh run and while its
# model is being reconstructed before a resume checkpoint is loaded.
if [[ "$STAGE" == "stage3" ]]; then
  TEACHER_CKPT=${TEACHER_CKPT:-${INIT_CKPT:-$(latest_checkpoint "$WORK_ROOT/stage2b/weight" 'best_mIoU*.pth')}}
  echo "Teacher checkpoint: $TEACHER_CKPT"
  ARGS+=(--cfg-options "model.teacher_ckpt=$TEACHER_CKPT")
fi

mkdir -p "$WORK_DIR"
if [[ "$RESUME" != "1" && -f "$WORK_DIR/last_checkpoint" ]]; then
  echo "Existing run found in $WORK_DIR." >&2
  echo "Use RESUME=1, or choose a new WORK_DIR for a separate experiment." >&2
  exit 1
fi
echo "Stage: $STAGE"
echo "GPUs: $GPUS"
echo "Work directory: $WORK_DIR"
echo "Resume: $RESUME"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'python tools/train.py'
  printf ' %q' "${ARGS[@]}"
  [[ "$GPUS" -gt 1 ]] && printf ' --launcher pytorch'
  printf '\n'
  exit 0
fi

[[ -d "$DINOV3_CHECKPOINT" ]] || {
  echo "DINOv3 checkpoint directory not found: $DINOV3_CHECKPOINT" >&2
  exit 1
}
if [[ "$STAGE" == "stage1" ]]; then
  [[ -d "$LLVIP_ROOT" ]] || {
    echo "LLVIP directory not found: $LLVIP_ROOT" >&2
    exit 1
  }
else
  [[ -d "$MFNET_ROOT" ]] || {
    echo "MFNet directory not found: $MFNET_ROOT" >&2
    exit 1
  }
fi

FREE_KB=$(df -Pk "$WORK_DIR" | awk 'NR==2 {print $4}')
if [[ "$MIN_FREE_GB" -gt 0 && "$FREE_KB" -lt $((MIN_FREE_GB * 1024 * 1024)) ]]; then
  echo "Less than ${MIN_FREE_GB} GiB is free under $WORK_DIR." >&2
  echo "Free disk space or explicitly set MIN_FREE_GB=0 to bypass this guard." >&2
  exit 1
fi

if [[ "$GPUS" -eq 1 ]]; then
  python tools/train.py "${ARGS[@]}"
else
  torchrun --standalone --nproc_per_node="$GPUS" tools/train.py \
    "${ARGS[@]}" --launcher pytorch
fi
