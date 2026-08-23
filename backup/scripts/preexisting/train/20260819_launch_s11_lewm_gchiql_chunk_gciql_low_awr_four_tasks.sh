#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/ogbench}"
TRAIN_SCRIPT="${OGBENCH_ROOT}/scripts/train/20260819_train_s11_lewm_gchiql_chunk_gciql_low_awr_s100k.sh"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

[[ -f "${TRAIN_SCRIPT}" ]] || { echo "ERROR: training script not found: ${TRAIN_SCRIPT}" >&2; exit 1; }

tasks=(tworoom reacher pusht cube)
gpus=(0 1 2 3)

for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  gpu="${gpus[$i]}"
  session="s11-gchiql-chunk-gciql-low-${task}"
  exp_name="GCHIQLChunk_GCIQLLowAWR_ogbench_lewm_${task}_k5_sg10_bs256_s100k_seed0_lalpha3_lexp09_hexp07_aug05_${RUN_STAMP}"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi

  tmux new-session -d -s "${session}" \
    env CUDA_VISIBLE_DEVICES="${gpu}" RUN_STAMP="${RUN_STAMP}" EXP_NAME="${exp_name}" \
    bash "${TRAIN_SCRIPT}" "${task}"
  echo "Started ${exp_name} on GPU ${gpu} in tmux ${session}"
done
