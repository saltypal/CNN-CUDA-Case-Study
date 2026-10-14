#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PYTHON="${HPC_CNN_PYTHON:-/home/Satya/.venvs/hpc-cnn/bin/python}"
export PYTHONPATH="${TASK_ROOT}/src"
export PATH="/usr/local/cuda-12.9/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd -- "${TASK_ROOT}"
exec "${TASK_PYTHON}" -m hpc_cnn.run "$@"
