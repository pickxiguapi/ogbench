#!/usr/bin/env bash
set -euo pipefail

# A800 node4：四卡并行评测 shared-all GCIQL-Chunk mode 初始化的 K10
# LatentPathFlow subgoal-guided LeWM CEM。Policy 与 CEM 共用同一个 predicted
# latent subgoal；自动 H2、RH1、MoH、CEM300x30、goal/budget=25/50。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

GPU_IDS=${GPU_IDS:-"0 1 2 3"}
NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
POLICY_SEED=${POLICY_SEED:-777}
GOAL_OFFSET_STEPS=${GOAL_OFFSET_STEPS:-25}
EVAL_BUDGET=${EVAL_BUDGET:-50}
NUM_SAMPLES=${NUM_SAMPLES:-1}
POLICY_STEPS=100000
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror}
SUBGOAL_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10
EVAL_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks
OUTPUT_ROOT=${OUTPUT_ROOT:-$EVAL_ROOT/20260901_gciql_chunk_all_sd${POLICY_SEED}_latent_path_flow_hist3_k10_ns${NUM_SAMPLES}_guided_moh_cem300x30_h2_rh1_g${GOAL_OFFSET_STEPS}_b${EVAL_BUDGET}_ep${NUM_EVAL}_seed${EVAL_SEED}}
TMP_ROOT=/data-training/yyf/ogbench-lewm-policy-runs/tmp/gciql-chunk-all-k10-subgoal-guided

tasks=(cube pusht reacher tworoom)
read -r -a gpus <<< "$GPU_IDS"
if (( ${#gpus[@]} != ${#tasks[@]} )); then
  echo "GPU_IDS must contain exactly four whitespace-separated GPU IDs." >&2
  exit 2
fi

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

cd "$OGBENCH_ROOT/impls"
pids=()
for i in "${!tasks[@]}"; do
  task=${tasks[$i]}
  policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
  output_dir="$OUTPUT_ROOT/$task"
  task_tmp="$TMP_ROOT/sd${POLICY_SEED}/$task"
  mkdir -p "$output_dir" "$task_tmp"

  TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" eval_lewm_4tasks.py \
    --task="$task" --controller=lewm_cem --policy-guidance=mode --use-subgoal \
    --data-root="$LEWM_DATA_ROOT" \
    --lewm-checkpoint="${lewm_checkpoints[$i]}" \
    --policy-checkpoint-dir="$policy_dir" --policy-checkpoint-step="$POLICY_STEPS" \
    --latent-subgoal-checkpoint="${subgoal_checkpoints[$i]}" \
    --num-samples="$NUM_SAMPLES" \
    --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
    --goal-offset-steps="$GOAL_OFFSET_STEPS" --eval-budget="$EVAL_BUDGET" \
    --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
    --cem-num-samples=300 --cem-iterations=30 --cem-topk=30 --cem-var-scale=1.0 \
    --cem-cost-mode=moh \
    --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
exit "$failed"
