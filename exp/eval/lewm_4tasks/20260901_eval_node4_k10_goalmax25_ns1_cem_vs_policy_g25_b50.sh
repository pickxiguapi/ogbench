#!/usr/bin/env bash
set -euo pipefail

# A800 node4：新 goalmax25 hist3 K10 predictor 的 25/50 正式评测。
# GPU0–3 跑纯 LeWM CEM；GPU4–7 跑 shared-all AWR seed777 Policy Mode。
# 两组均为 single-sample subgoal、MoH、自动 H2、J5、300/top-30、
# action_block=5、RH1、50 episodes、evaluation seed42；Q/V 不参与。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
POLICY_SEED=${POLICY_SEED:-777}
POLICY_STEPS=100000
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror}
SUBGOAL_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10-goalmax25
EVAL_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks
TMP_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/tmp/k10-goalmax25-ns1-g25-b50

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

run_group() {
  local gpu_ids=$1
  local policy_guidance=$2
  local method_tag=$3
  local output_root="$EVAL_ROOT/20260901_k10_goalmax25_ns1_${method_tag}_moh_cem300x5_h2_rh1_g25_b50_ep${NUM_EVAL}_seed${EVAL_SEED}"
  local -a gpus
  read -r -a gpus <<< "$gpu_ids"

  local -a pids=()
  for i in "${!tasks[@]}"; do
    local task=${tasks[$i]}
    local output_dir="$output_root/$task"
    local task_tmp="$TMP_ROOT/$method_tag/$task"
    local -a policy_args=()
    if [[ "$policy_guidance" == mode ]]; then
      local policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
      policy_args=(
        --policy-checkpoint-dir="$policy_dir"
        --policy-checkpoint-step="$POLICY_STEPS"
        --guidance-population-size=0
        --guidance-temperature=1.0
        --guidance-elite-size=8
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
        --task="$task" --controller=lewm_cem \
        --policy-guidance="$policy_guidance" --use-subgoal \
        "${policy_args[@]}" \
        --data-root="$LEWM_DATA_ROOT" \
        --lewm-checkpoint="${lewm_checkpoints[$i]}" \
        --latent-subgoal-checkpoint="${subgoal_checkpoints[$i]}" \
        --num-samples=1 \
        --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
        --goal-offset-steps=25 --eval-budget=50 \
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

run_group "0 1 2 3" none lewm_cem &
pid_cem=$!
run_group "4 5 6 7" mode actor_cem_mode_sd777 &
pid_policy=$!

failed=0
if ! wait "$pid_cem"; then failed=1; fi
if ! wait "$pid_policy"; then failed=1; fi
exit "$failed"
