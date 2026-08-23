#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/root/data/yyf/ogbench-new}"
DASHBOARD_ROOT="${DASHBOARD_ROOT:-/root/data/yyf/experiment-dashboard}"
EVAL_SCRIPT="${OGBENCH_ROOT}/scripts/eval/20260818_eval_yb_hiql_chunk_two_v_visual_s500k_seed42.sh"

[[ -f "${DASHBOARD_ROOT}/scripts/recorded_run.sh" ]] || { echo "ERROR: experiment recorder not found" >&2; exit 1; }
[[ -f "${EVAL_SCRIPT}" ]] || { echo "ERROR: evaluation Bash not found" >&2; exit 1; }

tasks=(cube-single cube-double scene)
gpus=(0 1 4)
run_ids=(EXP-021-YB-CUBE-SINGLE-R02 EXP-021-YB-CUBE-DOUBLE-R02 EXP-021-YB-SCENE-R02)
env_tags=(cube_single cube_double scene)

for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  gpu="${gpus[$i]}"
  run_id="${run_ids[$i]}"
  env_tag="${env_tags[$i]}"
  exp_name="HIQLChunkTwoV_ogbench_visual_${env_tag}_play_bs256_k5_sg10_aH3_aL3_exp07_lr3e4_aug05_s500k_seed0"
  session="exp021-eval-hiqlchunk2v-${task}-s500k-seed42"
  payload="{\"algorithm\":\"hiql_chunk_two_v\",\"environment\":\"${env_tag}\",\"checkpoint_step\":500000,\"chunk_size\":5,\"evaluation_seed\":42,\"eval_episodes_per_task\":50,\"evaluation_protocol\":\"ogbench_builtin_visual_play\"}"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: duplicate tmux session: ${session}" >&2
    exit 1
  fi

  tmux new-session -d -s "${session}" -c "${OGBENCH_ROOT}" \
    env CUDA_VISIBLE_DEVICES="${gpu}" EXPERIMENT_EXTRA_PAYLOAD_JSON="${payload}" \
    bash "${DASHBOARD_ROOT}/scripts/recorded_run.sh" \
      EXP-021 "${exp_name}" "${run_id}" --eval-only bash "${EVAL_SCRIPT}" "${task}"
  echo "started ${session}: GPU ${gpu}, run=${run_id}"
done
