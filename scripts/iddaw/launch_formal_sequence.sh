#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$ROOT/remote_logs/iddaw"
mkdir -p "$LOG_DIR"

EPOCHS="${EPOCHS:-25}"
DEVICE="${DEVICE:-0}"

if [[ $# -gt 0 ]]; then
  MODES=("$@")
else
  MODES=(rgb nir rgbnir)
fi

echo "[queue] root=$ROOT"
echo "[queue] dataset_root=${IDDAW_YOLO_ROOT:-${IDDAW_FOG_YOLO_ROOT:-unset}}"
echo "[queue] epochs=$EPOCHS device=$DEVICE"
echo "[queue] modes=${MODES[*]}"

for mode in "${MODES[@]}"; do
  echo "[queue] starting ${mode} at $(date +%F_%T)"
  bash "$ROOT/scripts/iddaw/launch_nohup_train.sh" "$mode" "$EPOCHS" "$DEVICE"

  pid_file="$LOG_DIR/latest_${mode}.pid"
  while true; do
    if [[ ! -e "$pid_file" ]]; then
      sleep 5
      continue
    fi

    pid="$(cat "$pid_file")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      sleep 60
    else
      echo "[queue] finished ${mode} at $(date +%F_%T)"
      break
    fi
  done
done

echo "[queue] all requested modes finished at $(date +%F_%T)"
