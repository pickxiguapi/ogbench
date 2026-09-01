#!/usr/bin/env bash
set -euo pipefail

# A800 node4：100/200 下比较 goalmax25 与 uniform-full hist3 K10。
# 两组均由 seed777 Policy Mode 引导，Policy goal 可通过
# GUIDANCE_GOAL_MODE=final|subgoal 切换；CEM cost 始终使用 predicted
# z_{t+10}。MoH、自动 H2、RH1、J5、300/top-30、action_block=5、
# single-sample subgoal、50 episodes、seed42；Q/V 不参与。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
POLICY_SEED=${POLICY_SEED:-777}
GUIDANCE_GOAL_MODE=${GUIDANCE_GOAL_MODE:-final}
case "$GUIDANCE_GOAL_MODE" in
  final) GUIDANCE_TAG=finalgoal ;;
  subgoal) GUIDANCE_TAG=subgoal ;;
  *) echo "GUIDANCE_GOAL_MODE must be final or subgoal" >&2; exit 2 ;;
esac
POLICY_STEPS=100000
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror}
GOALMAX_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10-goalmax25
UNIFORM_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10
EVAL_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks
TMP_ROOT="/data-training/yyf/ogbench-lewm-policy-runs/tmp/k10-policy-${GUIDANCE_TAG}-g100-b200"

tasks=(cube pusht reacher tworoom)
lewm_checkpoints=(
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
)
goalmax_checkpoints=(
  "$GOALMAX_ROOT/latent_pathflow_cube_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$GOALMAX_ROOT/latent_pathflow_pusht_lewm666_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$GOALMAX_ROOT/latent_pathflow_reacher_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$GOALMAX_ROOT/latent_pathflow_tworoom_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
)
uniform_checkpoints=(
  "$UNIFORM_ROOT/latent_pathflow_cube_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$UNIFORM_ROOT/latent_pathflow_pusht_lewm666_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$UNIFORM_ROOT/latent_pathflow_reacher_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$UNIFORM_ROOT/latent_pathflow_tworoom_lewm3072_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
)

run_generator() {
  local gpu_ids=$1
  local generator_tag=$2
  local checkpoint_array_name=$3
  local -n checkpoints=$checkpoint_array_name
  local output_root="$EVAL_ROOT/20260901_k10_${generator_tag}_ns1_actor_cem_mode_${GUIDANCE_TAG}_sd${POLICY_SEED}_moh_cem300x5_h2_rh1_g100_b200_ep${NUM_EVAL}_seed${EVAL_SEED}"
  local -a gpus
  read -r -a gpus <<< "$gpu_ids"

  local -a pids=()
  for i in "${!tasks[@]}"; do
    local task=${tasks[$i]}
    local policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
    local output_dir="$output_root/$task"
    local task_tmp="$TMP_ROOT/$generator_tag/$task"
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
        --guidance-goal-mode="$GUIDANCE_GOAL_MODE" \
        --guidance-population-size=0 --guidance-temperature=1.0 --guidance-elite-size=8 \
        --data-root="$LEWM_DATA_ROOT" \
        --lewm-checkpoint="${lewm_checkpoints[$i]}" \
        --policy-checkpoint-dir="$policy_dir" --policy-checkpoint-step="$POLICY_STEPS" \
        --latent-subgoal-checkpoint="${checkpoints[$i]}" --num-samples=1 \
        --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
        --goal-offset-steps=100 --eval-budget=200 \
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

run_generator "0 1 2 3" goalmax25 goalmax_checkpoints &
pid_goalmax=$!
run_generator "4 5 6 7" uniform_future uniform_checkpoints &
pid_uniform=$!

failed=0
if ! wait "$pid_goalmax"; then failed=1; fi
if ! wait "$pid_uniform"; then failed=1; fi
exit "$failed"
