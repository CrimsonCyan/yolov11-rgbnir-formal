#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: show_latest_run.sh <rgb|nir|rgbnir>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
META_FILE="$ROOT/remote_logs/iddaw/latest_${MODE}.meta"

if [[ ! -e "$META_FILE" ]]; then
  echo "latest metadata not found: $META_FILE" >&2
  exit 1
fi

cat "$META_FILE"
