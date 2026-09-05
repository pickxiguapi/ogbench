#!/usr/bin/env bash
set -euo pipefail

# Re-run a small sample of the real general-generator H50 policy evaluation and
# capture its exact closed-loop subgoal events.  The policy observes real env
# frames every step, refreshes the general K10 target every five steps, plans
# two action blocks with CEM, and executes only the first block before replanning.
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

TORCH_PYTHON_BIN=${TORCH_PYTHON_BIN:-/data-training/yyf/envs/latent-geometry/bin/python}
NUM_EVAL=${NUM_EVAL:-4}
EVAL_SEED=${EVAL_SEED:-42}
GOAL_OFFSET=${GOAL_OFFSET:-50}
EVAL_BUDGET=${EVAL_BUDGET:-100}
POLICY_SEED=${POLICY_SEED:-777}
POLICY_STEPS=100000
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror}
SUBGOAL_ROOT=${SUBGOAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10}
DECODER_ROOT=${DECODER_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/lewm-visual-decoder/20260905_mixed666_3072_official_cnn_image_decoder_aligned_epoch10_snapshot}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/lewm-visual-decoder-eval/20260906_general_h50_closed_loop_trace}
MODE=${MODE:-launch}
mkdir -p "$OUTPUT_ROOT/logs"

run_task() {
  local task=$1 gpu=$2 lewm_seed=$3 lewm_checkpoint=$4
  local policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
  local subgoal="$SUBGOAL_ROOT/latent_pathflow_${task}_lewm${lewm_seed}_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  local decoder="$DECODER_ROOT/$task/best.pt"
  local output_dir="$OUTPUT_ROOT/$task"
  local trace_dir="$output_dir/trace"
  local figure_dir="$output_dir/figures"
  mkdir -p "$output_dir" "$trace_dir" "$figure_dir" "$output_dir/tmp"
  test -s "$lewm_checkpoint"
  test -s "$subgoal"
  test -s "$decoder"

  cd "$OGBENCH_ROOT/impls"
  TMPDIR="$output_dir/tmp" CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" eval_lewm_4tasks.py \
    --task="$task" --controller=lewm_cem --policy-guidance=mode \
    --use-subgoal --guidance-goal-mode=final \
    --guidance-population-size=0 --guidance-temperature=1.0 \
    --guidance-elite-size=8 \
    --data-root="$LEWM_DATA_ROOT" --lewm-checkpoint="$lewm_checkpoint" \
    --policy-checkpoint-dir="$policy_dir" --policy-checkpoint-step="$POLICY_STEPS" \
    --latent-subgoal-checkpoint="$subgoal" --num-samples=1 \
    --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
    --goal-offset-steps="$GOAL_OFFSET" --eval-budget="$EVAL_BUDGET" \
    --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
    --cem-num-samples=300 --cem-iterations=5 --cem-topk=30 \
    --cem-var-scale=1.0 --cem-cost-mode=moh \
    --trace-dir="$trace_dir" --output="$output_dir/result.json" \
    >"$OUTPUT_ROOT/logs/$task.log" 2>&1

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$TORCH_PYTHON_BIN" render_lewm_subgoal_trace.py \
    --task="$task" --trace-dir="$trace_dir" \
    --decoder-checkpoint="$decoder" --output-dir="$figure_dir" \
    --goal-offset="$GOAL_OFFSET" --waypoint-step=10 --display-stride=5 \
    >>"$OUTPUT_ROOT/logs/$task.log" 2>&1
}

case "$MODE" in
  worker)
    run_task "${TASK:?TASK is required}" "${GPU_ID:?GPU_ID is required}" \
      "${LEWM_SEED:?LEWM_SEED is required}" "${LEWM_CHECKPOINT:?LEWM_CHECKPOINT is required}"
    ;;
  launch)
    specs=(
      "cube:4:3072:/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"
      "pusht:5:666:/data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack"
      "reacher:6:3072:/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"
      "tworoom:7:3072:/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"
    )
    for spec in "${specs[@]}"; do
      IFS=: read -r task gpu lewm_seed lewm_checkpoint <<<"$spec"
      session="lewm-h50-trace-$task"
      tmux has-session -t "$session" 2>/dev/null && { echo "Session exists: $session" >&2; exit 3; }
      printf -v worker_cmd '%q ' env MODE=worker TASK="$task" GPU_ID="$gpu" \
        LEWM_SEED="$lewm_seed" LEWM_CHECKPOINT="$lewm_checkpoint" \
        PYTHON_BIN="$PYTHON_BIN" TORCH_PYTHON_BIN="$TORCH_PYTHON_BIN" \
        LEWM_DATA_ROOT="$LEWM_DATA_ROOT" POLICY_ROOT="$POLICY_ROOT" \
        SUBGOAL_ROOT="$SUBGOAL_ROOT" DECODER_ROOT="$DECODER_ROOT" \
        OUTPUT_ROOT="$OUTPUT_ROOT" NUM_EVAL="$NUM_EVAL" EVAL_SEED="$EVAL_SEED" \
        GOAL_OFFSET="$GOAL_OFFSET" EVAL_BUDGET="$EVAL_BUDGET" \
        POLICY_SEED="$POLICY_SEED" \
        bash exp/eval/lewm_4tasks/20260906_eval_node4_general_h50_closed_loop_visual_trace.sh
      tmux new-session -d -s "$session" -c "$OGBENCH_ROOT" "$worker_cmd"
    done
    echo "launched real closed-loop H50 traces on GPUs 4,5,6,7"
    ;;
  status)
    tmux list-sessions 2>/dev/null | grep 'lewm-h50-trace-' || true
    for log in "$OUTPUT_ROOT"/logs/*.log; do
      [[ -f "$log" ]] || continue
      echo "===== $log ====="
      tail -n 12 "$log"
    done
    ;;
  *)
    echo "MODE must be worker, launch, or status" >&2
    exit 2
    ;;
esac
