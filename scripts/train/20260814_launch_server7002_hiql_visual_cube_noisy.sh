#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/yyf/ogbench}"
RECORDER_ROOT="${RECORDER_ROOT:-/home/yyf/yyf/experiment-dashboard}"
EXP_ID="EXP-020"

tasks=(single double triple)
gpus=(0 1 2)
run_ids=(EXP-020-R01 EXP-020-R02 EXP-020-R03)

[[ -f "${RECORDER_ROOT}/scripts/recorded_run.sh" ]] || { echo "ERROR: experiment recorder unavailable" >&2; exit 1; }
command -v tmux >/dev/null 2>&1 || { echo "ERROR: tmux unavailable" >&2; exit 1; }

cd "$OGBENCH_ROOT"
for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  gpu="${gpus[$i]}"
  run_id="${run_ids[$i]}"
  session="exp020-hiql-visual-cube-${task}-noisy"
  exp_name="HIQL_ogbench_visual_cube_${task}_noisy_impalasmall_bs256_s500k_seed0_sg10_official"

  tmux has-session -t "$session" 2>/dev/null && { echo "ERROR: tmux session exists: ${session}" >&2; exit 1; }
  printf -v command \
    'cd %q && CUDA_VISIBLE_DEVICES=%q EXPERIMENT_EXTRA_PAYLOAD_JSON=%q bash %q %q %q %q -- bash %q %q' \
    "$OGBENCH_ROOT" "$gpu" \
    "{\"environment\":\"visual-cube-${task}-noisy-v0\",\"algorithm\":\"HIQL\",\"train_steps\":500000,\"batch_size\":256,\"seed\":0,\"subgoal_steps\":10}" \
    "${RECORDER_ROOT}/scripts/recorded_run.sh" "$EXP_ID" "$exp_name" "$run_id" \
    "${OGBENCH_ROOT}/scripts/train/20260814_train_server7002_hiql_visual_cube_noisy.sh" "$task"
  tmux new-session -d -s "$session" -c "$OGBENCH_ROOT" "$command"
  echo "Started ${run_id}: visual-cube-${task}-noisy-v0 on GPU ${gpu} (tmux: ${session})"
done
