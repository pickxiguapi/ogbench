#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/ogbench}"
RUNS_ROOT="${RUNS_ROOT:-/data/yyf/H-LeWM/ogbench-runs}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EVAL_SCRIPT="${OGBENCH_ROOT}/scripts/eval/20260820_eval_s11_lewm_hiql_chunk_gciql_low_awr_task_s100k_seed42.sh"

tasks=(tworoom reacher pusht cube)
gpus=(0 1 2 3)

[[ -f "${EVAL_SCRIPT}" ]] || { echo "ERROR: evaluation script not found: ${EVAL_SCRIPT}" >&2; exit 1; }

for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  gpu="${gpus[$i]}"
  session="s11-eval-hiql-chunk-${task}"
  run_group="gchiql-chunk-gciql-low-awr-lewm-${task}-k5-sg10-bs256-s100k"
  checkpoint_root="${RUNS_ROOT}/${task}/OGBench/${run_group}"
  exp_name="HIQLChunk_GCIQLLowAWR_ogbench_lewm_${task}_k5_sg10_s100k_direct_g25_b50_seed42"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi

  command="while ! find '${checkpoint_root}' -mindepth 2 -maxdepth 2 -type f -name 'params_100000.pkl' -size +0c | grep -q .; do sleep '${POLL_SECONDS}'; done; CUDA_VISIBLE_DEVICES='${gpu}' EXP_NAME='${exp_name}' bash '${EVAL_SCRIPT}' '${task}'"
  tmux new-session -d -s "${session}" "${command}"
  echo "queued ${task}: tmux=${session}, gpu=${gpu}"
done
