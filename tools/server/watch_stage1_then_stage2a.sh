#!/usr/bin/env bash
set -euo pipefail

ROOT=${QWSEG_ROOT:-/root/autodl-tmp/code/QWSEG}
WORK_ROOT=${WORK_ROOT:-$ROOT/work_dirs/dino_ts}
POLL_SECONDS=${POLL_SECONDS:-60}
GPUS=${GPUS:-1}
STAGE2A_MAX_EPOCHS=${STAGE2A_MAX_EPOCHS:-200}
STAGE1_EXIT="$WORK_ROOT/stage1/train.exit"
STAGE2A_EXIT="$WORK_ROOT/stage2a/train.exit"

mkdir -p "$WORK_ROOT/stage2a"

printf 'Waiting for Stage 1 completion marker: %s\n' "$STAGE1_EXIT"
while [[ ! -f "$STAGE1_EXIT" ]]; do
  sleep "$POLL_SECONDS"
done

stage1_status=$(tr -d '[:space:]' < "$STAGE1_EXIT")
if [[ "$stage1_status" != "0" ]]; then
  printf 'Stage 1 failed with exit code %s; Stage 2A will not start.\n' "$stage1_status" >&2
  exit 1
fi

best_stage1=$(find "$WORK_ROOT/stage1/weight" -type f \
  -name 'best_stage1_align_loss*.pth' -print -quit)
if [[ -z "$best_stage1" ]]; then
  printf 'Stage 1 completed but no best checkpoint was found.\n' >&2
  exit 1
fi

if [[ -f "$STAGE2A_EXIT" ]]; then
  printf 'Stage 2A completion marker already exists: %s\n' "$STAGE2A_EXIT" >&2
  exit 1
fi

printf 'Stage 1 succeeded. Starting Stage 2A from %s\n' "$best_stage1"
set +e
GPUS="$GPUS" MAX_EPOCHS="$STAGE2A_MAX_EPOCHS" \
  bash "$ROOT/tools/server/train_dino_ts.sh" stage2a
stage2a_status=$?
set -e
printf '%s\n' "$stage2a_status" > "$STAGE2A_EXIT"
exit "$stage2a_status"
