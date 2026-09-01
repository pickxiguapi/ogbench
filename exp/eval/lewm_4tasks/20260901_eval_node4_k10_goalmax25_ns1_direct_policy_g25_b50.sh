#!/usr/bin/env bash
set -euo pipefail

# A800 node4：四卡并行评测 goalmax25 hist3 K10 + shared-all AWR seed777
# Direct Policy，不使用 LeWM CEM/Q/V。25/50、single-sample subgoal、
# action_block=5、50 episodes、evaluation seed42。
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
OUTPUT_ROOT="$EVAL_ROOT/20260901_k10_goalmax25_ns1_direct_policy_sd${POLICY_SEED}_g25_b50_ep${NUM_EVAL}_seed${EVAL_SEED}"
TMP_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/tmp/k10-goalmax25-ns1-direct-policy

tasks=(cube pusht reacher tworoom)
gpus=(0 1 2 3)
subgoal_checkpoints=(
  "$SUBGOAL_ROOT/latent_pathflow_cube_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_pusht_lewm666_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_reacher_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$SUBGOAL_ROOT/latent_pathflow_tworoom_lewm3072_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
)

pids=()
for i in "${!tasks[@]}"; do
  task=${tasks[$i]}
  policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
  output_dir="$OUTPUT_ROOT/$task"
  task_tmp="$TMP_ROOT/$task"
  mkdir -p "$output_dir" "$task_tmp"

  (
    cd "$OGBENCH_ROOT/impls"
    TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" eval_lewm_4tasks.py \
      --task="$task" --controller=direct_policy --policy-guidance=none --use-subgoal \
      --data-root="$LEWM_DATA_ROOT" \
      --policy-checkpoint-dir="$policy_dir" --policy-checkpoint-step="$POLICY_STEPS" \
      --latent-subgoal-checkpoint="${subgoal_checkpoints[$i]}" --num-samples=1 \
      --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
      --goal-offset-steps=25 --eval-budget=50 --action-block=5 \
      --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
exit "$failed"
