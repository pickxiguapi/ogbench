#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/ogbench}"
DATA_ROOT="${DATA_ROOT:-/data/yyf/H-LeWM/datasets}"
OGBENCH_PYTHON="${OGBENCH_PYTHON:-/data/yyf/H-LeWM/envs/ogbench/bin/python}"

[[ -x "${OGBENCH_PYTHON}" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -f "${OGBENCH_ROOT}/scripts/convert_lewm_hdf5_to_lance.py" ]] || {
  echo "ERROR: converter not found under ${OGBENCH_ROOT}" >&2
  exit 1
}

convert_dataset() {
  local dataset="$1"
  local source_path target_path complete_marker
  source_path="${DATA_ROOT}/${dataset}.h5"
  target_path="${DATA_ROOT}/${dataset}.lance"
  complete_marker="${target_path}/.conversion_complete"
  [[ -s "${source_path}" ]] || { echo "ERROR: dataset not found: ${source_path}" >&2; exit 1; }
  if [[ -f "${complete_marker}" ]]; then
    echo "SKIP: Lance dataset already exists: ${target_path}"
    return 0
  fi
  echo "Converting ${source_path} -> ${target_path}"
  (
    cd "${OGBENCH_ROOT}"
    "${OGBENCH_PYTHON}" scripts/convert_lewm_hdf5_to_lance.py \
      "${source_path}" "${target_path}"
  )
  touch "${complete_marker}"
  echo "DONE: ${target_path}"
}

datasets=(tworoom reacher pusht_expert_train cube_single_expert)
pids=()
for dataset in "${datasets[@]}"; do
  convert_dataset "${dataset}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
