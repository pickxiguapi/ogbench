#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/dzb/ogbench}"
DATA_ROOT="${DATA_ROOT:-/data/dzb/stablewm-data/datasets}"

cd "${OGBENCH_ROOT}"
"${OGBENCH_ROOT}/.venv/bin/python" scripts/convert_lewm_hdf5_to_lance.py \
  "${DATA_ROOT}/pusht_expert_train.h5" \
  "${DATA_ROOT}/pusht_expert_train.lance"
