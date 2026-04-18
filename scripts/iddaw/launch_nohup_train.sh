#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: launch_nohup_train.sh <rgb|nir|rgbnir|input_fusion|light_gate|bifpn_only|attention_only|full_proposed|full_proposed_residual|full_proposed_residual_v2> [epochs] [device]}"
EPOCHS="${2:-1}"
DEVICE="${3:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lym/anaconda3/envs/visnir-exp/bin/python}"
DEFAULT_DATA_ROOT="/home/lym/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir"
export IDDAW_YOLO_ROOT="${IDDAW_YOLO_ROOT:-${IDDAW_FOG_YOLO_ROOT:-$DEFAULT_DATA_ROOT}}"
export PYTHONUNBUFFERED=1

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
  --device "$DEVICE"
)

nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"

cat >"$META_FILE" <<EOF
mode=$MODE
epochs=$EPOCHS
device=$DEVICE
pid=$PID
log_file=$LOG_FILE
python_bin=$PYTHON_BIN
dataset_root=$IDDAW_YOLO_ROOT
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
