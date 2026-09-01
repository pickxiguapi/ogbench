#!/usr/bin/env bash
set -euo pipefail

# A800 node4：新 aligned-future hist3 K10 predictor + shared-all AWR seed777
# Policy Mode 初始化 LeWM CEM，Q/V 不参与。评测 Goal/Budget=25/50、50/100、
# 75/150；single-sample subgoal、MoH、自动 H2、J5、300/top-30、
# action_block=5、RH1、50 episodes、evaluation seed42。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
POLICY_SEED=${POLICY_SEED:-777}
POLICY_STEPS=100000
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror}
SUBGOAL_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10-aligned-future
EVAL_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks
TMP_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/tmp/k10-aligned-future-ns1-actor-mode

tasks=(cube pusht reacher tworoom)
lewm_checkpoints=(
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
)
subgoal_checkpoints=(
  "$SUBGOAL_ROOT/latent_pathflow_cube_lewm3072_hist3_sg10_ab5_goalstride5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_pusht_lewm666_hist3_sg10_ab5_goalstride5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_reacher_lewm3072_hist3_sg10_ab5_goalstride5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_tworoom_lewm3072_hist3_sg10_ab5_goalstride5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
)

run_setting() {
  local gpu_ids=$1
  local goal_offset=$2
  local eval_budget=$3
  local output_root="$EVAL_ROOT/20260901_k10_aligned_future_ns1_actor_cem_mode_sd${POLICY_SEED}_moh_cem300x5_h2_rh1_g${goal_offset}_b${eval_budget}_ep${NUM_EVAL}_seed${EVAL_SEED}"
  local -a gpus
  read -r -a gpus <<< "$gpu_ids"

  local -a pids=()
  for i in "${!tasks[@]}"; do
    local task=${tasks[$i]}
    local policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
    local output_dir="$output_root/$task"
    local task_tmp="$TMP_ROOT/g${goal_offset}_b${eval_budget}/$task"
    mkdir -p "$output_dir" "$task_tmp"

    (
      cd "$OGBENCH_ROOT/impls"
      TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
      LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
      "$PYTHON_BIN" eval_lewm_4tasks.py \
        --task="$task" --controller=lewm_cem --policy-guidance=mode --use-subgoal \
        --guidance-population-size=0 --guidance-temperature=1.0 --guidance-elite-size=8 \
        --data-root="$LEWM_DATA_ROOT" \
        --lewm-checkpoint="${lewm_checkpoints[$i]}" \
        --policy-checkpoint-dir="$policy_dir" --policy-checkpoint-step="$POLICY_STEPS" \
        --latent-subgoal-checkpoint="${subgoal_checkpoints[$i]}" \
        --num-samples=1 \
        --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
        --goal-offset-steps="$goal_offset" --eval-budget="$eval_budget" \
        --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
        --cem-num-samples=300 --cem-iterations=5 --cem-topk=30 --cem-var-scale=1.0 \
        --cem-cost-mode=moh \
        --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1
    ) &
    pids+=("$!")
  done

  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  return "$failed"
}

run_setting "0 1 2 3" 25 50 &
pid_25=$!
run_setting "4 5 6 7" 50 100 &
pid_50=$!

failed=0
if ! wait "$pid_25"; then failed=1; fi
if ! wait "$pid_50"; then failed=1; fi
if (( failed )); then exit "$failed"; fi

run_setting "0 1 2 3" 75 150
