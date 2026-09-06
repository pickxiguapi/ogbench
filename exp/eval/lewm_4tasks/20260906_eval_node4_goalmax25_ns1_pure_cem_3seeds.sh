#!/usr/bin/env bash
set -euo pipefail

# H25 single-variable ablation of the canonical LeWM++ result: remove the
# policy checkpoint and every policy-guidance mechanism, while retaining the
# goalmax25 LatentPathFlow subgoal generator and the complete CEM setup.
# Fixed settings: mixed LeWM checkpoints, FlowPath ns1, MoH, H2/RH1/J5,
# CEM300x5, goal offset 25, budget 50, 50 episodes, eval seeds 0/1/42.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="$SCRIPT_DIR/20260905_eval_node4_goalmax25_ns1_policy_combinations_seed42.sh"

EVAL_SEEDS=${EVAL_SEEDS:-"0 1 42"}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
NUM_EVAL=${NUM_EVAL:-50}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks}
TMP_BASE=${TMP_BASE:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260906-goalmax25-ns1-pure-cem}

read -r -a eval_seeds <<< "$EVAL_SEEDS"
read -r -a gpus <<< "$GPU_IDS"
if (( ${#gpus[@]} != 4 && ${#gpus[@]} != 8 )); then
  echo "GPU_IDS must contain exactly four or eight whitespace-separated GPU IDs." >&2
  exit 2
fi

seeds_per_batch=$((${#gpus[@]} / 4))

run_seed() {
  local eval_seed=$1
  shift
  local seed_gpus=("$@")
  local output_root="$EVAL_ROOT/20260906_goalmax25_ns1_pure_cem_no_policy_moh_cem300x5_h2_rh1_g25_b50_ep${NUM_EVAL}_seed${eval_seed}"
  local tmp_root="$TMP_BASE/ep${NUM_EVAL}/seed${eval_seed}"

  echo "RUN pure CEM without policy guidance: seed=$eval_seed GPUs=${seed_gpus[*]}"
  EVAL_SEED="$eval_seed" \
  NUM_EVAL="$NUM_EVAL" \
  GPU_IDS="${seed_gpus[*]}" \
  RUN_VARIANTS=zero_init \
  SKIP_COMPLETED="$SKIP_COMPLETED" \
  GOAL_OFFSET_STEPS=25 \
  EVAL_BUDGET=50 \
  CEM_COST_MODE=moh \
  OUTPUT_ROOT="$output_root" \
  TMP_ROOT="$tmp_root" \
    bash "$BASE_SCRIPT"
}

failed=0
for ((base=0; base<${#eval_seeds[@]}; base+=seeds_per_batch)); do
  pids=()
  for ((slot=0; slot<seeds_per_batch && base+slot<${#eval_seeds[@]}; slot++)); do
    gpu_offset=$((slot * 4))
    run_seed "${eval_seeds[$((base + slot))]}" "${gpus[@]:gpu_offset:4}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
done

exit "$failed"
