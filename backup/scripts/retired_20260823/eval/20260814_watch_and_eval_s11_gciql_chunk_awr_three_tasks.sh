#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/ogbench}"
RUNS_ROOT="${RUNS_ROOT:-/data/yyf/H-LeWM/ogbench-runs}"
POLL_SECONDS="${POLL_SECONDS:-60}"

tasks=(tworoom reacher pusht)
gpus=(0 1 2)

for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  gpu="${gpus[$i]}"
  session="s11-eval-after-train-${task}"
  train_pattern="^/data/yyf/H-LeWM/envs/ogbench/bin/python main.py .*--save_dir=${RUNS_ROOT}/${task}"
  command="while pgrep -f '${train_pattern}' >/dev/null; do sleep ${POLL_SECONDS}; done; ckpt=\$(find '${RUNS_ROOT}/${task}/OGBench/lewm-${task}-gciql-chunk-awr-k5-bs256-s100k-s11' -type f -name params_100000.pkl | sort | tail -n 1); test -s \"\${ckpt}\"; CUDA_VISIBLE_DEVICES=${gpu} bash '${OGBENCH_ROOT}/scripts/eval/20260814_record_s11_gciql_chunk_awr_task_eval.sh' '${task}'"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
  tmux new-session -d -s "${session}" "${command}"
  echo "Watching ${task}; evaluation will use GPU ${gpu}; tmux: ${session}"
done
