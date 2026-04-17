#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: stop_latest_train.sh <rgb|nir|rgbnir>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PID_FILE="$ROOT/remote_logs/iddaw/latest_${MODE}.pid"

if [[ ! -e "$PID_FILE" ]]; then
  echo "latest pid file not found: $PID_FILE" >&2
  exit 1
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "stopped pid=$PID"
else
  echo "process not running: pid=$PID"
fi
