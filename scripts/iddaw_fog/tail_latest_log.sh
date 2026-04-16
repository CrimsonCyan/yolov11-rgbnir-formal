#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: tail_latest_log.sh <rgb|nir|rgbnir>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_FILE="$ROOT/remote_logs/iddaw_fog/latest_${MODE}.stdout.log"

if [[ ! -e "$LOG_FILE" ]]; then
  echo "latest log not found: $LOG_FILE" >&2
  exit 1
fi

tail -f "$LOG_FILE"
