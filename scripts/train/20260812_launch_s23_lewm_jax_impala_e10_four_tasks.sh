#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/home/dzb/ogbench"
DASHBOARD_ROOT="/home/dzb/experiment-dashboard"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

[[ -f "${DASHBOARD_ROOT}/scripts/recorded_run.sh" ]] || { echo "ERROR: dashboard recorder unavailable" >&2; exit 1; }

tasks=(cube pusht reacher tworoom)
gpus=(2 3 4 5)
names=(
  LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
)

for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  gpu="${gpus[$i]}"
  session="s23-lewm-jax-impala-e10-${task}"
  run_id="EXP-014-s23-${task}-${RUN_STAMP}"
  train_script="${OGBENCH_ROOT}/scripts/train/20260819_train_s23_lewm_jax_impala_task_e10.sh"
  eval_script="${OGBENCH_ROOT}/scripts/eval/20260812_eval_s23_lewm_jax_impala_${task}_e10_cem300x30.sh"

  tmux has-session -t "${session}" 2>/dev/null && { echo "ERROR: tmux session exists: ${session}" >&2; exit 1; }
  tmux new-session -d -s "${session}" \
    env CUDA_VISIBLE_DEVICES="${gpu}" \
      EXPERIMENT_EXTRA_PAYLOAD_JSON="{\"model_backend\":\"lewm_impala_small\",\"encoder\":\"impala_small\",\"dataset_backend\":\"jpeg95_lance\",\"epochs\":10,\"evaluation\":\"dataset_goal_cem300x30\"}" \
      bash "${DASHBOARD_ROOT}/scripts/recorded_run.sh" \
        EXP-014 "${names[$i]}_${RUN_STAMP}" "${run_id}" \
        --train bash "${train_script}" "${task}" --eval bash "${eval_script}"
  echo "Started ${run_id} on GPU ${gpu} in tmux ${session}"
done
