#!/usr/bin/env bash
set -euo pipefail

# A800 node4：在 LeWM-4Tasks 的 25/50 协议下做 planner objective × policy
# guidance 消融。固定同一套 frozen LeWM、uniform-future K10 LatentPathFlow、
# H2/RH1、CEM300x5、action block 5、50 episodes，并配对使用 evaluation
# seeds 0/1/42。objective 比较 last（Terminal）、moh 与 path_mean；guidance
# 比较 none（zero-init CEM）、mode、mode_anchor、population64_t03。
# policy 始终看 final goal，Q/V/verifier 均不参与；另跑一次 final-goal
# policy-only（不按 objective 重复）。默认只用当前空闲的 GPU4-7。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

EVAL_SEEDS=${EVAL_SEEDS:-"0 1 42"}
GPU_IDS=${GPU_IDS:-"4 5 6 7"}
NUM_EVAL=${NUM_EVAL:-50}
POLICY_SEED=${POLICY_SEED:-777}
POLICY_STEPS=100000
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror}
SUBGOAL_ROOT=${SUBGOAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260904-planner-cost-policy-guidance-3seed-g25-b50}

source "$OGBENCH_ROOT/scripts/client_env.sh"

tasks=(cube pusht reacher tworoom)
lewm_checkpoints=(
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
)
subgoal_checkpoints=(
  "$SUBGOAL_ROOT/latent_pathflow_cube_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_pusht_lewm666_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_reacher_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_tworoom_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
)

read -r -a eval_seeds <<< "$EVAL_SEEDS"
read -r -a gpus <<< "$GPU_IDS"
if (( ${#gpus[@]} != 4 )); then
  echo "GPU_IDS must contain exactly four whitespace-separated GPU IDs." >&2
  exit 2
fi

run_tasks() {
  local eval_seed=$1
  local objective=$2
  local guidance=$3
  local population_size=$4
  local temperature=$5
  local tag=$6
  local output_root="$EVAL_ROOT/20260904_k10_ns1_${tag}_${objective}_cem300x5_h2_rh1_g25_b50_ep${NUM_EVAL}_seed${eval_seed}"
  local -a pids=()

  for i in "${!tasks[@]}"; do
    local task=${tasks[$i]}
    local output_dir="$output_root/$task"
    local task_tmp="$TMP_ROOT/seed${eval_seed}/${objective}/${tag}/$task"
    local policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
    local -a policy_args=()
    if [[ "$guidance" != none ]]; then
      policy_args+=(
        --policy-checkpoint-dir="$policy_dir"
        --policy-checkpoint-step="$POLICY_STEPS"
      )
    fi
    mkdir -p "$output_dir" "$task_tmp"

    (
      cd "$OGBENCH_ROOT/impls"
      TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
      LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
      "$PYTHON_BIN" eval_lewm_4tasks.py \
        --task="$task" --controller=lewm_cem --policy-guidance="$guidance" --use-subgoal \
        --guidance-goal-mode=final \
        --guidance-population-size="$population_size" \
        --guidance-temperature="$temperature" --guidance-elite-size=8 \
        --data-root="$LEWM_DATA_ROOT" \
        --lewm-checkpoint="${lewm_checkpoints[$i]}" \
        "${policy_args[@]}" \
        --latent-subgoal-checkpoint="${subgoal_checkpoints[$i]}" --num-samples=1 \
        --num-eval="$NUM_EVAL" --seed="$eval_seed" \
        --goal-offset-steps=25 --eval-budget=50 \
        --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
        --cem-num-samples=300 --cem-iterations=5 --cem-topk=30 --cem-var-scale=1.0 \
        --cem-cost-mode="$objective" \
        --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1
    ) &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  return "$failed"
}

run_policy_only() {
  local eval_seed=$1
  local output_root="$EVAL_ROOT/20260904_finalgoal_policy_only_sd${POLICY_SEED}_g25_b50_ep${NUM_EVAL}_seed${eval_seed}"
  local -a pids=()

  for i in "${!tasks[@]}"; do
    local task=${tasks[$i]}
    local output_dir="$output_root/$task"
    local task_tmp="$TMP_ROOT/seed${eval_seed}/policy_only/$task"
    local policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
    mkdir -p "$output_dir" "$task_tmp"
    (
      cd "$OGBENCH_ROOT/impls"
      TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
      LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
      "$PYTHON_BIN" eval_lewm_4tasks.py \
        --task="$task" --controller=direct_policy --policy-guidance=none \
        --data-root="$LEWM_DATA_ROOT" \
        --policy-checkpoint-dir="$policy_dir" --policy-checkpoint-step="$POLICY_STEPS" \
        --num-eval="$NUM_EVAL" --seed="$eval_seed" \
        --goal-offset-steps=25 --eval-budget=50 --action-block=5 \
        --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1
    ) &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  return "$failed"
}

failed=0
for eval_seed in "${eval_seeds[@]}"; do
  if ! run_policy_only "$eval_seed"; then failed=1; fi
  for objective in last moh path_mean; do
    if ! run_tasks "$eval_seed" "$objective" none 0 1.0 zero_init; then failed=1; fi
    if ! run_tasks "$eval_seed" "$objective" mode 0 1.0 policy_mode; then failed=1; fi
    if ! run_tasks "$eval_seed" "$objective" mode_anchor 0 1.0 policy_mode_anchor; then failed=1; fi
    if ! run_tasks "$eval_seed" "$objective" population 64 0.3 policy_population64_t03; then failed=1; fi
  done
done
exit "$failed"
