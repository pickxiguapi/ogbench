#!/usr/bin/env bash
set -euo pipefail

# 正文复现总入口：分别运行三组 H25 配对消融，并将各组 full rerun 保持独立。
# 可选运行 H25/H50/H75/H100 主评测；每个 generator family 使用独立目录和配置核验。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

: "${LEWM_DATA_ROOT:?Set LEWM_DATA_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
TASKS=${TASKS:-"cube pusht reacher tworoom"}
GPU_IDS=${GPU_IDS:-"0 1 2 3"}
RUN_LONG_HORIZON=${RUN_LONG_HORIZON:-1}
RUN_BASELINES=${RUN_BASELINES:-1}
read -r -a tasks <<< "$TASKS"
read -r -a gpus <<< "$GPU_IDS"
if (( ${#tasks[@]} != ${#gpus[@]} )); then
  echo "TASKS and GPU_IDS must contain the same number of entries." >&2
  exit 2
fi

lookup() {
  local prefix=$1
  local task=$2
  local suffix=$3
  local name="${prefix}_${task^^}_${suffix}"
  printf '%s' "${!name:-}"
}

run_setting() {
  local group=$1
  local horizon=$2
  local seed=$3
  local variant=$4
  local prior_mode=${5:-policy_mode}
  local family=GOALMAX25
  if (( horizon > 25 )); then family=GENERAL; fi
  local pids=()
  local failed=0
  for i in "${!tasks[@]}"; do
    local task=${tasks[$i]}
    local lewm policy flow
    lewm=$(lookup LEWM "$task" CHECKPOINT)
    policy=$(lookup POLICY "$task" CHECKPOINT_DIR)
    flow=$(lookup "$family" "$task" CHECKPOINT)
    (
      export CUDA_VISIBLE_DEVICES=${gpus[$i]}
      export TASK="$task"
      export VARIANT="$variant"
      export ACTION_PRIOR_MODE="$prior_mode"
      export GENERATOR_TYPE=latent_path_flow
      export EXPERIMENT_GROUP="$group"
      export GOAL_OFFSET_STEPS="$horizon"
      export EVAL_BUDGET=$((horizon * 2))
      export EVAL_SEED="$seed"
      export LEWM_CHECKPOINT="$lewm"
      export POLICY_CHECKPOINT_DIR="$policy"
      export SUBGOAL_GENERATOR_CHECKPOINT="$flow"
      bash "$SCRIPT_DIR/evaluate.sh"
    ) &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  return "$failed"
}

failed=0
for seed in 0 1 42; do
  run_setting h25_policy 25 "$seed" full || failed=1
  run_setting h25_policy 25 "$seed" no_action_prior zero || failed=1
  run_setting h25_subgoal 25 "$seed" full || failed=1
  run_setting h25_subgoal 25 "$seed" no_subgoal || failed=1
done
for seed in 0 1 666; do
  run_setting h25_temporal_score 25 "$seed" full || failed=1
  run_setting h25_temporal_score 25 "$seed" no_moh || failed=1
done

if [[ "$RUN_LONG_HORIZON" == 1 ]]; then
  for seed in 0 1 42; do
    for horizon in 25 50 75 100; do
      run_setting long_horizon "$horizon" "$seed" full || failed=1
    done
  done
fi
if [[ "$RUN_BASELINES" == 1 ]]; then
  for seed in 0 1 42; do
    for horizon in 25 50 75 100; do
      run_setting baselines "$horizon" "$seed" lewm zero || failed=1
      run_setting baselines "$horizon" "$seed" gciql_chunk policy_mode || failed=1
    done
  done
fi
exit "$failed"
