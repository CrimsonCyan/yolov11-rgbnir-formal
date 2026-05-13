#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: launch_nohup_faster_rcnn.sh <faster_rcnn_rgb_8cls_detectable640|faster_rcnn_nir_8cls_detectable640> [epochs] [device] [resume_ckpt]}"
EPOCHS="${2:-50}"
DEVICE="${3:-${DEVICE:-0}}"
RESUME_CKPT="${4:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data1/lvyanhu/miniconda3/envs/visnir-exp/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/data1/lvyanhu/miniconda3/envs/visnir-exp/bin/torchrun}"
DEFAULT_DATA_ROOT="/data1/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir_8cls_personmerge_traffic_detectable640"
DETECTABLE640_BASENAME="iddaw_all_weather_full_yolov11_rgbnir_8cls_personmerge_traffic_detectable640"

case "$MODE" in
  faster_rcnn_rgb_8cls_detectable640)
    MODALITY="rgb"
    ;;
  faster_rcnn_nir_8cls_detectable640)
    MODALITY="nir"
    ;;
  *)
    echo "Unknown Faster R-CNN mode: $MODE" >&2
    exit 2
    ;;
esac

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-/data1/lvyanhu/.cache/torch}"
export DATASET_ROOT="${DATASET_ROOT:-$DEFAULT_DATA_ROOT}"
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "ERROR: DATASET_ROOT does not exist: $DATASET_ROOT" >&2
  exit 2
fi
DATASET_ROOT="$(cd "$DATASET_ROOT" && pwd -P)"
export DATASET_ROOT
if [[ "$(basename "$DATASET_ROOT")" != "$DETECTABLE640_BASENAME" ]]; then
  echo "ERROR: Faster R-CNN 8cls baselines must use detectable640 dataset: $DETECTABLE640_BASENAME" >&2
  echo "Got DATASET_ROOT=$DATASET_ROOT" >&2
  exit 2
fi
export IMGSZ="${IMGSZ:-640}"
export BATCH_PER_GPU="${BATCH_PER_GPU:-16}"
export WORKERS="${WORKERS:-4}"
export PRETRAINED="${PRETRAINED:-true}"
export AMP="${AMP:-true}"
export LR="${LR:-0.02}"
export MOMENTUM="${MOMENTUM:-0.9}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
export LR_MILESTONES="${LR_MILESTONES:-30,40}"
export LR_GAMMA="${LR_GAMMA:-0.1}"
export WARMUP_ITERS="${WARMUP_ITERS:-1000}"
export WARMUP_FACTOR="${WARMUP_FACTOR:-0.001}"
export CONF="${CONF:-0.001}"
export PR_CONF="${PR_CONF:-0.25}"
export IOU="${IOU:-0.7}"
export AREA_IMGSZ="${AREA_IMGSZ:-640}"
export EVAL_BACKEND="${EVAL_BACKEND:-coco}"
export PROFILE_GFLOPS="${PROFILE_GFLOPS:-1}"
export MAX_TRAIN_IMAGES="${MAX_TRAIN_IMAGES:-0}"
export MAX_VAL_IMAGES="${MAX_VAL_IMAGES:-0}"

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
export WANDB_PROJECT="${WANDB_PROJECT:-iddaw-rgbnir-formal}"
export WANDB_GROUP="${WANDB_GROUP:-external_faster_rcnn}"
export WANDB_TAGS="${WANDB_TAGS:-${MODE},external,faster-rcnn,8-class-personmerge-traffic}"

LOG_DIR="$ROOT/remote_logs/external"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${MODE}_e${EPOCHS}_${STAMP}"
LOG_FILE="$LOG_DIR/${RUN_NAME}.stdout.log"
PID_FILE="$LOG_DIR/${RUN_NAME}.pid"
META_FILE="$LOG_DIR/${RUN_NAME}.meta"
LATEST_LOG="$LOG_DIR/latest_${MODE}.stdout.log"
LATEST_PID="$LOG_DIR/latest_${MODE}.pid"
LATEST_META="$LOG_DIR/latest_${MODE}.meta"
PROJECT="$ROOT/runs/external/faster_rcnn"

TRAIN_ARGS=(
  "$ROOT/scripts/external/train_faster_rcnn.py"
  --modality "$MODALITY"
  --dataset-root "$DATASET_ROOT"
  --epochs "$EPOCHS"
  --imgsz "$IMGSZ"
  --batch-per-gpu "$BATCH_PER_GPU"
  --workers "$WORKERS"
  --project "$PROJECT"
  --name "$RUN_NAME"
  --pretrained "$PRETRAINED"
  --amp "$AMP"
  --lr "$LR"
  --momentum "$MOMENTUM"
  --weight-decay "$WEIGHT_DECAY"
  --lr-milestones "$LR_MILESTONES"
  --lr-gamma "$LR_GAMMA"
  --warmup-iters "$WARMUP_ITERS"
  --warmup-factor "$WARMUP_FACTOR"
  --conf "$CONF"
  --pr-conf "$PR_CONF"
  --iou "$IOU"
  --area-imgsz "$AREA_IMGSZ"
  --eval-backend "$EVAL_BACKEND"
  --max-train-images "$MAX_TRAIN_IMAGES"
  --max-val-images "$MAX_VAL_IMAGES"
)

if [[ "$PROFILE_GFLOPS" != "1" && "$PROFILE_GFLOPS" != "true" && "$PROFILE_GFLOPS" != "True" ]]; then
  TRAIN_ARGS+=(--no-profile-gflops)
fi

if [[ -n "$RESUME_CKPT" ]]; then
  TRAIN_ARGS+=(--resume "$RESUME_CKPT")
fi

DEVICE_CLEAN="${DEVICE// /}"
if [[ "$DEVICE_CLEAN" == "cpu" ]]; then
  CMD=("$PYTHON_BIN" "${TRAIN_ARGS[@]}" --device cpu)
else
  export CUDA_VISIBLE_DEVICES="$DEVICE_CLEAN"
  if [[ "$DEVICE_CLEAN" == *","* ]]; then
    IFS=',' read -r -a GPU_IDS <<<"$DEVICE_CLEAN"
    NPROC="${#GPU_IDS[@]}"
    CMD=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" "${TRAIN_ARGS[@]}" --device cuda)
  else
    CMD=("$PYTHON_BIN" "${TRAIN_ARGS[@]}" --device cuda)
  fi
fi

nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"

cat >"$META_FILE" <<EOF
mode=$MODE
modality=$MODALITY
epochs=$EPOCHS
device=$DEVICE
cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}
resume_ckpt=$RESUME_CKPT
pid=$PID
log_file=$LOG_FILE
python_bin=$PYTHON_BIN
torchrun_bin=$TORCHRUN_BIN
torch_home=$TORCH_HOME
dataset_root=$DATASET_ROOT
project=$PROJECT
run_name=$RUN_NAME
wandb_enabled=$WANDB_ENABLED
wandb_console=${WANDB_CONSOLE:-}
wandb_project=${WANDB_PROJECT:-}
wandb_entity=${WANDB_ENTITY:-}
wandb_group=${WANDB_GROUP:-}
wandb_tags=${WANDB_TAGS:-}
imgsz=$IMGSZ
batch_per_gpu=$BATCH_PER_GPU
workers=$WORKERS
pretrained=$PRETRAINED
amp=$AMP
lr=$LR
momentum=$MOMENTUM
weight_decay=$WEIGHT_DECAY
lr_milestones=$LR_MILESTONES
lr_gamma=$LR_GAMMA
warmup_iters=$WARMUP_ITERS
warmup_factor=$WARMUP_FACTOR
conf=$CONF
pr_conf=$PR_CONF
iou=$IOU
area_imgsz=$AREA_IMGSZ
eval_backend=$EVAL_BACKEND
profile_gflops=$PROFILE_GFLOPS
max_train_images=$MAX_TRAIN_IMAGES
max_val_images=$MAX_VAL_IMAGES
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
