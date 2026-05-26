#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data1/lvyanhu/miniconda3/envs/visnir-exp/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  CONDA_SH="/data1/lvyanhu/miniconda3/etc/profile.d/conda.sh"
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    echo "Conda profile script not found: ${CONDA_SH}" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate visnir-exp
  PYTHON_BIN="python"
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7860}"
DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-640}"
CONF="${CONF:-0.25}"
IOU="${IOU:-0.7}"
MAX_DET="${MAX_DET:-300}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WEIGHTS="${WEIGHTS:-${ROOT_DIR}/runs/IDD_AW/MGFDet/weights/best.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/runs/web_detect}"
GRADIO_TEMP_DIR="${GRADIO_TEMP_DIR:-${ROOT_DIR}/runs/web_detect/gradio_tmp}"

mkdir -p "${GRADIO_TEMP_DIR}" "${OUTPUT_ROOT}"
export GRADIO_TEMP_DIR
export TMPDIR="${TMPDIR:-${GRADIO_TEMP_DIR}}"
export TEMP="${TEMP:-${GRADIO_TEMP_DIR}}"
export TMP="${TMP:-${GRADIO_TEMP_DIR}}"

if ! "${PYTHON_BIN}" - <<'PY'
import gradio  # noqa: F401
PY
then
  echo "gradio is not installed in visnir-exp. Install with: ${PYTHON_BIN} -m pip install gradio" >&2
  exit 1
fi

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Weight file not found: ${WEIGHTS}" >&2
  exit 1
fi

mkdir -p remote_logs/web_demo
TS="$(date +%Y%m%d_%H%M%S)"
LOG="remote_logs/web_demo/rgbnir_detect_web_${TS}.log"

nohup "${PYTHON_BIN}" apps/rgbnir_detect_web.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  --imgsz "${IMGSZ}" \
  --conf "${CONF}" \
  --iou "${IOU}" \
  --max-det "${MAX_DET}" \
  --batch-size "${BATCH_SIZE}" \
  --output-root "${OUTPUT_ROOT}" \
  --weights "${WEIGHTS}" \
  > "${LOG}" 2>&1 &

PID="$!"
echo "RGB-NIR detection web demo started."
echo "PID: ${PID}"
echo "Log: ${ROOT_DIR}/${LOG}"
echo "Bind: http://${HOST}:${PORT}"
echo "Recommended tunnel: ssh -L ${PORT}:127.0.0.1:${PORT} lyh"
