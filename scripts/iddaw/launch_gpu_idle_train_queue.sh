#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$ROOT/remote_logs/iddaw"
mkdir -p "$LOG_DIR"

EPOCHS="${EPOCHS:-100}"
DEVICE="${DEVICE:-0,1}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-20}"
DATA_CACHE="${DATA_CACHE:-ram}"
OPTIMIZER="${OPTIMIZER:-Adam}"
LR0="${LR0:-0.01}"
PRETRAINED="${PRETRAINED:-true}"
SMALL_CENTER_GAIN="${SMALL_CENTER_GAIN:-0}"
SMALL_SCALE_GAIN="${SMALL_SCALE_GAIN:-0}"
GPU_TARGETS="${GPU_TARGETS:-0 1}"
GPU_CHECK_INTERVAL_SECONDS="${GPU_CHECK_INTERVAL_SECONDS:-300}"
GPU_CONFIRM_DELAY_SECONDS="${GPU_CONFIRM_DELAY_SECONDS:-300}"
QUEUE_GAP_SECONDS="${QUEUE_GAP_SECONDS:-120}"
QUEUE_RETRY_LIMIT="${QUEUE_RETRY_LIMIT:-2}"
QUEUE_RETRY_DELAY_SECONDS="${QUEUE_RETRY_DELAY_SECONDS:-300}"
GPU_IDLE_POWER_W="${GPU_IDLE_POWER_W:-50}"
GPU_IDLE_MEM_MB="${GPU_IDLE_MEM_MB:-1000}"

if [[ $# -gt 0 ]]; then
  MODES=("$@")
else
  MODES=(
    bifpn_only_light_nir_p2p5_oa_segmask_p2only_c256_r4_reduction1_yolo11s_8cls_personmerge_traffic
    bifpn_only_light_nir_p2p5_oa_segmask_fg050_p2only_c256_r4_reduction1_yolo11s_8cls_personmerge_traffic
    nir_p2p5_yolo11s_8cls_personmerge_traffic
    early_fusion_p2p5_yolo11s_8cls_personmerge_traffic
    rgb_p2p5_yolov8s_8cls_personmerge_traffic
    nir_p2p5_yolov8s_8cls_personmerge_traffic
    rgbnir_halfway_c3_p2p5_yolo11s_8cls_personmerge_traffic
  )
fi

gpu_idle_once() {
  local rows row gpu found index power mem
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu-idle] nvidia-smi not found" >&2
    return 1
  fi

  rows="$(nvidia-smi --query-gpu=index,power.draw,memory.used --format=csv,noheader,nounits)"
  for gpu in $GPU_TARGETS; do
    found=0
    while IFS=, read -r index power mem; do
      index="${index//[[:space:]]/}"
      power="${power//[[:space:]]/}"
      mem="${mem//[[:space:]]/}"
      if [[ "$index" != "$gpu" ]]; then
        continue
      fi
      found=1
      awk -v p="$power" -v m="$mem" -v pmax="$GPU_IDLE_POWER_W" -v mmax="$GPU_IDLE_MEM_MB" \
        'BEGIN { exit !((p + 0) < (pmax + 0) && (m + 0) < (mmax + 0)) }' || return 1
    done <<<"$rows"
    if [[ "$found" != "1" ]]; then
      echo "[gpu-idle] gpu $gpu not found" >&2
      return 1
    fi
  done
}

wait_for_gpu_idle() {
  echo "[gpu-idle] targets=[$GPU_TARGETS] power<${GPU_IDLE_POWER_W}W mem<${GPU_IDLE_MEM_MB}MB"
  while true; do
    if gpu_idle_once; then
      echo "[gpu-idle] first idle check passed at $(date +%F_%T); confirming after ${GPU_CONFIRM_DELAY_SECONDS}s"
      sleep "$GPU_CONFIRM_DELAY_SECONDS"
      if gpu_idle_once; then
        echo "[gpu-idle] confirmed idle at $(date +%F_%T)"
        return 0
      fi
      echo "[gpu-idle] confirm check failed at $(date +%F_%T); keep waiting"
    else
      echo "[gpu-idle] busy at $(date +%F_%T); recheck after ${GPU_CHECK_INTERVAL_SECONDS}s"
      sleep "$GPU_CHECK_INTERVAL_SECONDS"
    fi
  done
}

wait_for_mode_pid() {
  local mode="$1"
  local latest_pid="$LOG_DIR/latest_${mode}.pid"
  local latest_log="$LOG_DIR/latest_${mode}.stdout.log"
  local pid

  while [[ ! -e "$latest_pid" ]]; do
    sleep 5
  done
  pid="$(cat "$latest_pid")"
  echo "[queue] waiting mode=$mode pid=$pid"

  while [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done

  if [[ -e "$latest_log" ]] && grep -Ei "Traceback|RuntimeError|CUDA out of memory|shape mismatch|NaN" "$latest_log" >/dev/null; then
    echo "[queue] error pattern found in $latest_log; stop queue" >&2
    return 1
  fi
}

echo "[queue] root=$ROOT"
echo "[queue] epochs=$EPOCHS device=$DEVICE gap=${QUEUE_GAP_SECONDS}s"
echo "[queue] imgsz=$IMGSZ batch=$BATCH cache=$DATA_CACHE optimizer=$OPTIMIZER lr0=$LR0 pretrained=$PRETRAINED"
echo "[queue] small_loss=center_gain:$SMALL_CENTER_GAIN scale_gain:$SMALL_SCALE_GAIN"
echo "[queue] retry_limit=$QUEUE_RETRY_LIMIT retry_delay=${QUEUE_RETRY_DELAY_SECONDS}s"
echo "[queue] schema=8cls_personmerge_traffic"
echo "[queue] dataset policy: oa_segmask modes use segment labels; other modes use bbox labels"
echo "[queue] modes=${MODES[*]}"

for i in "${!MODES[@]}"; do
  mode="${MODES[$i]}"
  attempt=1
  while true; do
    wait_for_gpu_idle
    echo "[queue] starting $mode attempt=$attempt at $(date +%F_%T)"
    WANDB_ENABLED="${WANDB_ENABLED:-1}" IDDAW_CLASS_SCHEMA=8cls_personmerge_traffic \
      IMGSZ="$IMGSZ" BATCH="$BATCH" DATA_CACHE="$DATA_CACHE" OPTIMIZER="$OPTIMIZER" LR0="$LR0" \
      PRETRAINED="$PRETRAINED" SMALL_CENTER_GAIN="$SMALL_CENTER_GAIN" SMALL_SCALE_GAIN="$SMALL_SCALE_GAIN" \
      bash "$ROOT/scripts/iddaw/launch_nohup_train.sh" "$mode" "$EPOCHS" "$DEVICE"

    if wait_for_mode_pid "$mode"; then
      echo "[queue] finished $mode attempt=$attempt at $(date +%F_%T)"
      break
    fi

    if [[ "$attempt" -ge "$QUEUE_RETRY_LIMIT" ]]; then
      echo "[queue] mode=$mode failed after $attempt attempt(s); stop queue" >&2
      exit 1
    fi

    attempt=$((attempt + 1))
    echo "[queue] mode=$mode failed; retry attempt=$attempt after ${QUEUE_RETRY_DELAY_SECONDS}s"
    sleep "$QUEUE_RETRY_DELAY_SECONDS"
  done

  if [[ "$i" -lt $((${#MODES[@]} - 1)) ]]; then
    echo "[queue] sleeping ${QUEUE_GAP_SECONDS}s before next mode"
    sleep "$QUEUE_GAP_SECONDS"
  fi
done

echo "[queue] all requested modes finished at $(date +%F_%T)"
