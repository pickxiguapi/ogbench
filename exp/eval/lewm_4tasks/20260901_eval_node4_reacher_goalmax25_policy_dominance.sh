#!/usr/bin/env bash
set -euo pipefail

# A800 node4：只针对 Reacher 调整 goalmax25 K10 + seed777 policy 的权重。
# 八卡分别测试 direct policy、mode anchor、mode 的首 action-block 方差
# 0.05/0.1/0.2/0.3，以及首轮 300/300 policy population（T=0.1/0.3）。
# CEM 变体均保持 25/50、MoH、自动 H2、RH1、J5、300/top-30；
# 所有变体 50 episodes、evaluation seed42，Q/V 不参与。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
POLICY_SEED=${POLICY_SEED:-777}
POLICY_STEPS=100000
POLICY_DIR=/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror/gc4_reacher_all_n100000_b256_a0.0_sd777
LEWM_CHECKPOINT=/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
SUBGOAL_CHECKPOINT=/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10-goalmax25/latent_pathflow_reacher_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack
EVAL_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks
TMP_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/tmp/reacher-goalmax25-policy-dominance

run_direct_policy() {
  local gpu=$1
  local tag=direct_policy
  local output_dir="$EVAL_ROOT/20260901_reacher_k10_goalmax25_ns1_${tag}_sd${POLICY_SEED}_g25_b50_ep${NUM_EVAL}_seed${EVAL_SEED}/reacher"
  mkdir -p "$output_dir" "$TMP_ROOT/$tag"
  cd "$OGBENCH_ROOT/impls"
  TMPDIR="$TMP_ROOT/$tag" CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" eval_lewm_4tasks.py \
    --task=reacher --controller=direct_policy --policy-guidance=none --use-subgoal \
    --data-root="$LEWM_DATA_ROOT" \
    --policy-checkpoint-dir="$POLICY_DIR" --policy-checkpoint-step="$POLICY_STEPS" \
    --latent-subgoal-checkpoint="$SUBGOAL_CHECKPOINT" --num-samples=1 \
    --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
    --goal-offset-steps=25 --eval-budget=50 --action-block=5 \
    --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1
}

run_cem_variant() {
  local gpu=$1
  local tag=$2
  local guidance_mode=$3
  local population_size=$4
  local temperature=$5
  local first_block_std=$6
  local output_dir="$EVAL_ROOT/20260901_reacher_k10_goalmax25_ns1_${tag}_sd${POLICY_SEED}_moh_cem300x5_h2_rh1_g25_b50_ep${NUM_EVAL}_seed${EVAL_SEED}/reacher"
  local -a std_args=()
  if [[ -n "$first_block_std" ]]; then
    std_args+=(--guidance-first-block-std="$first_block_std")
  fi
  mkdir -p "$output_dir" "$TMP_ROOT/$tag"
  cd "$OGBENCH_ROOT/impls"
  TMPDIR="$TMP_ROOT/$tag" CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" eval_lewm_4tasks.py \
    --task=reacher --controller=lewm_cem --policy-guidance="$guidance_mode" --use-subgoal \
    --guidance-population-size="$population_size" \
    --guidance-temperature="$temperature" --guidance-elite-size=8 \
    "${std_args[@]}" \
    --data-root="$LEWM_DATA_ROOT" --lewm-checkpoint="$LEWM_CHECKPOINT" \
    --policy-checkpoint-dir="$POLICY_DIR" --policy-checkpoint-step="$POLICY_STEPS" \
    --latent-subgoal-checkpoint="$SUBGOAL_CHECKPOINT" --num-samples=1 \
    --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
    --goal-offset-steps=25 --eval-budget=50 \
    --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
    --cem-num-samples=300 --cem-iterations=5 --cem-topk=30 --cem-var-scale=1.0 \
    --cem-cost-mode=moh \
    --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1
}

pids=()
run_direct_policy 0 & pids+=("$!")
run_cem_variant 1 mode_anchor mode_anchor 0 1.0 "" & pids+=("$!")
run_cem_variant 2 mode_std005 mode 0 1.0 0.05 & pids+=("$!")
run_cem_variant 3 mode_std01 mode 0 1.0 0.1 & pids+=("$!")
run_cem_variant 4 mode_std02 mode 0 1.0 0.2 & pids+=("$!")
run_cem_variant 5 mode_std03 mode 0 1.0 0.3 & pids+=("$!")
run_cem_variant 6 population300_t01 population 300 0.1 "" & pids+=("$!")
run_cem_variant 7 population300_t03 population 300 0.3 "" & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
exit "$failed"
