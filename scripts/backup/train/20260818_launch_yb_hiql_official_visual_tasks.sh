#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/root/data/yyf/ogbench-new}"
DASHBOARD_ROOT="${DASHBOARD_ROOT:-/root/data/yyf/experiment-dashboard}"
TRAIN_SCRIPT="${OGBENCH_ROOT}/scripts/train/20260818_train_yb_hiql_official_visual_task_s500k.sh"
RUN_ATTEMPT="${RUN_ATTEMPT:-R01}"

[[ -f "${DASHBOARD_ROOT}/scripts/recorded_run.sh" ]] || { echo "ERROR: experiment recorder not found" >&2; exit 1; }
[[ -f "${TRAIN_SCRIPT}" ]] || { echo "ERROR: training Bash not found" >&2; exit 1; }

tasks=(cube-single cube-double cube-triple cube-quadruple scene)
env_tags=(cube_single cube_double cube_triple cube_quadruple scene)
gpus=(3 5 7 7 6)

for requested in "$@"; do
  found=0
  for task in "${tasks[@]}"; do
    [[ "${requested}" == "${task}" ]] && found=1
  done
  [[ ${found} -eq 1 ]] || { echo "ERROR: unknown requested task ${requested}" >&2; exit 2; }
done

for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  if [[ $# -gt 0 ]]; then
    selected=0
    for requested in "$@"; do
      [[ "${requested}" == "${task}" ]] && selected=1
    done
    [[ ${selected} -eq 1 ]] || continue
  fi

  env_tag="${env_tags[$i]}"
  gpu="${gpus[$i]}"
  run_id="EXP-022-YB-${env_tag^^}-${RUN_ATTEMPT}"
  run_id="${run_id//_/-}"
  exp_name="HIQL_ogbench_visual_${env_tag}_play_bs256_sg10_aH3_aL3_exp07_lr3e4_aug05_s500k_seed0_official"
  session="exp022-hiql-${task}"
  payload="{\"algorithm\":\"hiql\",\"environment\":\"${env_tag}\",\"batch_size\":256,\"subgoal_steps\":10,\"high_alpha\":3.0,\"low_alpha\":3.0,\"expectile\":0.7,\"learning_rate\":0.0003,\"augmentation_probability\":0.5,\"train_steps\":500000,\"seed\":0,\"encoder\":\"impala_small\",\"wandb_mode\":\"offline\",\"official_hyperparameters\":true}"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: duplicate tmux session: ${session}" >&2
    exit 1
  fi

  tmux new-session -d -s "${session}" -c "${OGBENCH_ROOT}" \
    env CUDA_VISIBLE_DEVICES="${gpu}" EXPERIMENT_EXTRA_PAYLOAD_JSON="${payload}" \
    bash "${DASHBOARD_ROOT}/scripts/recorded_run.sh" \
      EXP-022 "${exp_name}" "${run_id}" -- bash "${TRAIN_SCRIPT}" "${task}"
  echo "started ${session}: GPU ${gpu}, ${exp_name}, run=${run_id}"
done
