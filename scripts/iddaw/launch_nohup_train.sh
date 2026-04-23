#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: launch_nohup_train.sh <rgb|rgb_yolo11s|rgb_yolo11s_6cls_personmerge|rgb_rtdetr|nir|rgbnir|input_fusion|light_gate|bifpn_only|bifpn_only_yolo11s|bifpn_only_yolo11s_6cls_personmerge|attention_only|full_proposed|full_proposed_residual|full_proposed_residual_v2|full_proposed_residual_v2_yolo11s|full_proposed_residual_v2_yolo11s_6cls_personmerge|proposed_lite_yolo11s_6cls_personmerge> [epochs] [device] [resume_ckpt]}"
EPOCHS="${2:-1}"
DEVICE="${3:-0}"
RESUME_CKPT="${4:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lym/anaconda3/envs/visnir-exp/bin/python}"
DEFAULT_DATA_ROOT="/home/lym/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir"
DEFAULT_DATA_ROOT_6CLS_PERSONMERGE="/home/lym/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir_6cls_personmerge"
export IDDAW_CLASS_SCHEMA="${IDDAW_CLASS_SCHEMA:-6cls_personmerge}"
export IDDAW_YOLO_ROOT="${IDDAW_YOLO_ROOT:-${IDDAW_FOG_YOLO_ROOT:-$DEFAULT_DATA_ROOT}}"
if [[ "$MODE" == *_6cls_personmerge || "$IDDAW_CLASS_SCHEMA" == "6cls_personmerge" ]]; then
  export IDDAW_YOLO_ROOT_6CLS_PERSONMERGE="${IDDAW_YOLO_ROOT_6CLS_PERSONMERGE:-$DEFAULT_DATA_ROOT_6CLS_PERSONMERGE}"
fi
export PYTHONUNBUFFERED=1
export WANDB_ENABLED="${WANDB_ENABLED:-0}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export VAL_INTERVAL="${VAL_INTERVAL:-1}"
export IMGSZ="${IMGSZ:-640}"
export OPTIMIZER="${OPTIMIZER:-SGD}"
export BATCH="${BATCH:-}"
if [[ "$WANDB_ENABLED" == "1" ]]; then
  if [[ "$MODE" == *_6cls_personmerge || "$IDDAW_CLASS_SCHEMA" == "6cls_personmerge" ]]; then
    DATASET_TAG="6-class-personmerge"
  else
    DATASET_TAG="7-class"
  fi
  export WANDB_PROJECT="${WANDB_PROJECT:-iddaw-rgbnir-formal}"
  export WANDB_GROUP="${WANDB_GROUP:-iddaw_all_weather}"
  export WANDB_TAGS="${WANDB_TAGS:-${MODE},all-weather,${DATASET_TAG}}"
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
)

if [[ -n "$BATCH" ]]; then
  CMD+=(--batch "$BATCH")
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
