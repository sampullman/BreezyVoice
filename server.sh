#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${BREEZYVOICE_CONDA_ENV:-breezyvoice}"

export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
export BREEZYVOICE_DEVICE="${BREEZYVOICE_DEVICE:-cpu}"
export BREEZYVOICE_NUM_THREADS="${BREEZYVOICE_NUM_THREADS:-16}"
export BREEZYVOICE_ORT_INTRA_OP_THREADS="${BREEZYVOICE_ORT_INTRA_OP_THREADS:-8}"
export BREEZYVOICE_ORT_INTER_OP_THREADS="${BREEZYVOICE_ORT_INTER_OP_THREADS:-1}"

cd "${ROOT_DIR}"
exec python -m uvicorn api:app --host 127.0.0.1 --port 8080
