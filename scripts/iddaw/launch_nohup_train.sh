#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: launch_nohup_train.sh <mode> [epochs] [device] [resume_ckpt]}"
EPOCHS="${2:-1}"
DEVICE="${3:-0}"
RESUME_CKPT="${4:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data1/lvyanhu/miniconda3/envs/visnir-exp/bin/python}"
DEFAULT_DATA_ROOT="/data1/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir"
DEFAULT_DATA_ROOT_6CLS_PERSONMERGE="/data1/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir_6cls_personmerge"
export IDDAW_CLASS_SCHEMA="${IDDAW_CLASS_SCHEMA:-6cls_personmerge}"
export IDDAW_YOLO_ROOT="${IDDAW_YOLO_ROOT:-${IDDAW_FOG_YOLO_ROOT:-$DEFAULT_DATA_ROOT}}"
if [[ "$MODE" == *_6cls_personmerge || "$IDDAW_CLASS_SCHEMA" == "6cls_personmerge" ]]; then
  export IDDAW_YOLO_ROOT_6CLS_PERSONMERGE="${IDDAW_YOLO_ROOT_6CLS_PERSONMERGE:-$DEFAULT_DATA_ROOT_6CLS_PERSONMERGE}"
fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${WANDB_ENABLED+x}" ]]; then
  if [[ "$EPOCHS" -le 1 ]]; then
    export WANDB_ENABLED=0
  else
    export WANDB_ENABLED=1
  fi
else
  export WANDB_ENABLED
fi
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export VAL_INTERVAL="${VAL_INTERVAL:-1}"
export IMGSZ="${IMGSZ:-640}"
export OPTIMIZER="${OPTIMIZER:-SGD}"
export BATCH="${BATCH:-}"
export LR0="${LR0:-}"
export COS_LR="${COS_LR:-0}"
export PRETRAINED="${PRETRAINED:-true}"
export CLOSE_MOSAIC="${CLOSE_MOSAIC:-20}"
if [[ "$WANDB_ENABLED" == "1" ]]; then
  if [[ "$MODE" == *_6cls_personmerge || "$IDDAW_CLASS_SCHEMA" == "6cls_personmerge" ]]; then
    DATASET_TAG="6-class-personmerge"
  else
    DATASET_TAG="7-class"
  fi
  MODE_TAG="$MODE"
  if [[ ${#MODE_TAG} -gt 64 ]]; then
    MODE_TAG="${MODE_TAG:0:64}"
  fi
  export WANDB_PROJECT="${WANDB_PROJECT:-iddaw-rgbnir-formal}"
  export WANDB_GROUP="${WANDB_GROUP:-iddaw_all_weather}"
  export WANDB_TAGS="${WANDB_TAGS:-${MODE_TAG},all-weather,${DATASET_TAG}}"
fi

LOG_DIR="$ROOT/remote_logs/iddaw"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${MODE}_e${EPOCHS}_${STAMP}"
LOG_FILE="$LOG_DIR/${RUN_NAME}.stdout.log"
PID_FILE="$LOG_DIR/${RUN_NAME}.pid"
META_FILE="$LOG_DIR/${RUN_NAME}.meta"
LATEST_LOG="$LOG_DIR/latest_${MODE}.stdout.log"
LATEST_PID="$LOG_DIR/latest_${MODE}.pid"
LATEST_META="$LOG_DIR/latest_${MODE}.meta"

CMD=(
  "$PYTHON_BIN"
  "$ROOT/scripts/iddaw/run_experiment.py"
  --mode "$MODE"
  --task train
  --epochs "$EPOCHS"
  --imgsz "$IMGSZ"
  --val-interval "$VAL_INTERVAL"
  --device "$DEVICE"
  --optimizer "$OPTIMIZER"
  --pretrained "$PRETRAINED"
)

if [[ -n "$BATCH" ]]; then
  CMD+=(--batch "$BATCH")
fi

if [[ -n "$LR0" ]]; then
  CMD+=(--lr0 "$LR0")
fi

if [[ "$COS_LR" == "1" || "$COS_LR" == "true" || "$COS_LR" == "True" ]]; then
  CMD+=(--cos-lr)
fi

if [[ -n "$RESUME_CKPT" ]]; then
  CMD+=(--resume "$RESUME_CKPT")
fi

nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"

cat >"$META_FILE" <<EOF
mode=$MODE
epochs=$EPOCHS
device=$DEVICE
resume_ckpt=$RESUME_CKPT
resume_mode=$([[ -n "$RESUME_CKPT" ]] && echo ultralytics_native || echo none)
pid=$PID
log_file=$LOG_FILE
python_bin=$PYTHON_BIN
dataset_root=$IDDAW_YOLO_ROOT
dataset_root_6cls_personmerge=${IDDAW_YOLO_ROOT_6CLS_PERSONMERGE:-}
class_schema=$IDDAW_CLASS_SCHEMA
wandb_enabled=$WANDB_ENABLED
wandb_console=${WANDB_CONSOLE:-}
wandb_project=${WANDB_PROJECT:-}
wandb_entity=${WANDB_ENTITY:-}
wandb_group=${WANDB_GROUP:-}
wandb_tags=${WANDB_TAGS:-}
val_interval=$VAL_INTERVAL
imgsz=$IMGSZ
optimizer=$OPTIMIZER
batch=${BATCH:-}
lr0=${LR0:-}
cos_lr=$COS_LR
pretrained=$PRETRAINED
close_mosaic=$CLOSE_MOSAIC
started_at=$STAMP
command=${CMD[*]}
EOF

ln -sfn "$(basename "$LOG_FILE")" "$LATEST_LOG"
ln -sfn "$(basename "$PID_FILE")" "$LATEST_PID"
ln -sfn "$(basename "$META_FILE")" "$LATEST_META"

echo "started"
echo "pid=$PID"
echo "log_file=$LOG_FILE"
echo "meta_file=$META_FILE"
echo "tail_command=tail -f $LATEST_LOG"
