#!/usr/bin/env bash
set -euo pipefail

# A800 node4：搜索 LeWM++/Guided 的 shared-all GCIQL-Chunk-AWR policy seed
# 与 LatentPathFlow 推理样本数。Cube/Reacher/TwoRoom 固定 LeWM seed3072，
# PushT 固定 LeWM seed666；goalmax25 K10 FlowPath、policy 看 final goal、CEM
# 看 predicted K10，固定 MoH、H2/RH1/J5、CEM300x5、25/50、50 episodes。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

POLICY_SEEDS=${POLICY_SEEDS:-"777 789"}
NUM_SAMPLES_LIST=${NUM_SAMPLES_LIST:-"1 8"}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
POLICY_STEPS=100000
GOAL_OFFSET_STEPS=25
EVAL_BUDGET=50
CEM_ITERATIONS=5
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror}
SUBGOAL_ROOT=${SUBGOAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10-goalmax25}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks}
OUTPUT_ROOT=${OUTPUT_ROOT:-$EVAL_ROOT/20260903_goalmax25_policy_seed_search_moh_g25_b50_ep${NUM_EVAL}_evalseed${EVAL_SEED}}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260903-goalmax25-policy-seed-search}

source "$OGBENCH_ROOT/scripts/client_env.sh"

tasks=(cube pusht reacher tworoom)
lewm_checkpoints=(
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
)
subgoal_checkpoints=(
  "$SUBGOAL_ROOT/latent_pathflow_cube_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_pusht_lewm666_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_reacher_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_tworoom_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
)

read -r -a policy_seeds <<< "$POLICY_SEEDS"
read -r -a num_samples_values <<< "$NUM_SAMPLES_LIST"
read -r -a all_gpus <<< "$GPU_IDS"
if (( ${#all_gpus[@]} != 8 )); then
  echo "GPU_IDS must contain exactly eight whitespace-separated GPU IDs." >&2
  exit 2
fi

variant_seeds=()
variant_samples=()
for policy_seed in "${policy_seeds[@]}"; do
  for num_samples in "${num_samples_values[@]}"; do
    variant_seeds+=("$policy_seed")
    variant_samples+=("$num_samples")
  done
done

run_variant() {
  local policy_seed=$1
  local num_samples=$2
  shift 2
  local gpus=("$@")
  local pids=()

  for i in "${!tasks[@]}"; do
    local task=${tasks[$i]}
    local policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${policy_seed}"
    local output_dir="$OUTPUT_ROOT/policy_seed${policy_seed}_ns${num_samples}/$task"
    local task_tmp="$TMP_ROOT/policy_seed${policy_seed}_ns${num_samples}/$task"
    mkdir -p "$output_dir" "$task_tmp"

    TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" "$OGBENCH_ROOT/impls/eval_lewm_4tasks.py" \
      --task="$task" --controller=lewm_cem --policy-guidance=mode --use-subgoal \
      --guidance-goal-mode=final \
      --guidance-population-size=0 --guidance-temperature=1.0 --guidance-elite-size=8 \
      --data-root="$LEWM_DATA_ROOT" \
      --lewm-checkpoint="${lewm_checkpoints[$i]}" \
      --policy-checkpoint-dir="$policy_dir" --policy-checkpoint-step="$POLICY_STEPS" \
      --latent-subgoal-checkpoint="${subgoal_checkpoints[$i]}" \
      --num-samples="$num_samples" \
      --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
      --goal-offset-steps="$GOAL_OFFSET_STEPS" --eval-budget="$EVAL_BUDGET" \
      --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
      --cem-num-samples=300 --cem-iterations="$CEM_ITERATIONS" --cem-topk=30 --cem-var-scale=1.0 \
      --cem-cost-mode=moh \
      --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  return "$failed"
}

failed=0
for (( base=0; base<${#variant_seeds[@]}; base+=2 )); do
  batch_pids=()
  run_variant "${variant_seeds[$base]}" "${variant_samples[$base]}" \
    "${all_gpus[@]:0:4}" &
  batch_pids+=("$!")

  if (( base + 1 < ${#variant_seeds[@]} )); then
    run_variant "${variant_seeds[$((base + 1))]}" "${variant_samples[$((base + 1))]}" \
      "${all_gpus[@]:4:4}" &
    batch_pids+=("$!")
  fi

  for pid in "${batch_pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
done
exit "$failed"
