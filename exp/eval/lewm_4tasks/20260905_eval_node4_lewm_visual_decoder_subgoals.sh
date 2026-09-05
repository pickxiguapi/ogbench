#!/usr/bin/env bash
set -euo pipefail

# A800 node4：评测四任务官方 CNNImageDecoder，并渲染 maxgoal25 LatentPathFlow
# 在 held-out episode 上预测的 K5/K10 subgoal。固定 history3、goal offset 25、
# medoid-of-8；同时输出固定分位代表样本与明确标注的 best-case 样本。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
JAX_PYTHON_BIN=${JAX_PYTHON_BIN:-/data-training/yyf/ogbench/.venv/bin/python}
TORCH_PYTHON_BIN=${TORCH_PYTHON_BIN:-/data-training/yyf/envs/latent-geometry/bin/python}
DATA_ROOT=${DATA_ROOT:-/data-training/yyf/datasets/latent-geometry}
LATENT_ROOT=${LATENT_ROOT:-/data-training/yyf/datasets/lewm-latents}
DECODER_RUN_ROOT=${DECODER_RUN_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/lewm-visual-decoder}
DECODER_RUN_NAME=${DECODER_RUN_NAME:-20260905_mixed666_3072_official_cnn_image_decoder_fg10}
SUBGOAL_ROOT=${SUBGOAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10-goalmax25}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/lewm-visual-decoder-eval}
EVAL_NAME=${EVAL_NAME:-20260905_convdecoder_maxgoal25_ns8_h3_g25}
MODE=${MODE:-launch}
NUM_PAIRS=${NUM_PAIRS:-256}
NUM_SAMPLES=${NUM_SAMPLES:-8}
mkdir -p "$OUTPUT_ROOT/$EVAL_NAME/logs"

run_task() {
  local task=$1 gpu=$2 latent lance generator lewm_seed
  case "$task" in
    pusht)
      latent="$LATENT_ROOT/pusht_expert_train__lewm_s666_e10_z192.h5"
      lance="$DATA_ROOT/pusht_expert_train.lance"
      lewm_seed=666
      ;;
    cube)
      latent="$LATENT_ROOT/cube_single_expert__lewm_s3072_e10_z192.h5"
      lance="$DATA_ROOT/cube_single_expert.lance"
      lewm_seed=3072
      ;;
    reacher)
      latent="$LATENT_ROOT/reacher__lewm_s3072_e10_z192.h5"
      lance="$DATA_ROOT/reacher.lance"
      lewm_seed=3072
      ;;
    tworoom)
      latent="$LATENT_ROOT/tworoom__lewm_s3072_e10_z192.h5"
      lance="$DATA_ROOT/tworoom.lance"
      lewm_seed=3072
      ;;
    *) echo "Unknown task: $task" >&2; exit 2 ;;
  esac
  generator="$SUBGOAL_ROOT/latent_pathflow_${task}_lewm${lewm_seed}_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  local decoder="$DECODER_RUN_ROOT/$DECODER_RUN_NAME/$task/best.pt"
  test -s "$latent"
  test -d "$lance"
  test -s "$decoder"
  test -s "$generator"
  cd "$OGBENCH_ROOT/impls"
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 \
  "$JAX_PYTHON_BIN" eval_lewm_visual_decoder.py \
    --task="$task" \
    --latent-hdf5="$latent" \
    --lance-path="$lance" \
    --decoder-checkpoint="$decoder" \
    --subgoal-checkpoint="$generator" \
    --output-dir="$OUTPUT_ROOT/$EVAL_NAME/$task" \
    --num-pairs="$NUM_PAIRS" \
    --num-samples="$NUM_SAMPLES" \
    --goal-offset=25 --split-seed=0 --sampling-seed=42 \
    --phase=predict \
    2>&1 | tee "$OUTPUT_ROOT/$EVAL_NAME/logs/$task.log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
  "$TORCH_PYTHON_BIN" eval_lewm_visual_decoder.py \
    --task="$task" \
    --latent-hdf5="$latent" \
    --lance-path="$lance" \
    --decoder-checkpoint="$decoder" \
    --subgoal-checkpoint="$generator" \
    --output-dir="$OUTPUT_ROOT/$EVAL_NAME/$task" \
    --num-pairs="$NUM_PAIRS" \
    --num-samples="$NUM_SAMPLES" \
    --goal-offset=25 --split-seed=0 --sampling-seed=42 \
    --phase=render \
    2>&1 | tee -a "$OUTPUT_ROOT/$EVAL_NAME/logs/$task.log"
}

case "$MODE" in
  worker)
    run_task "${TASK:?TASK is required}" "${GPU_ID:?GPU_ID is required}"
    ;;
  launch)
    for spec in tworoom:4 pusht:5 cube:6 reacher:7; do
      task=${spec%%:*}; gpu=${spec##*:}; session="lewm-visdec-eval-$task"
      tmux has-session -t "$session" 2>/dev/null && { echo "Session exists: $session" >&2; exit 3; }
      tmux new-session -d -s "$session" \
        "cd '$OGBENCH_ROOT' && MODE=worker TASK='$task' GPU_ID='$gpu' DECODER_RUN_NAME='$DECODER_RUN_NAME' EVAL_NAME='$EVAL_NAME' bash exp/eval/lewm_4tasks/20260905_eval_node4_lewm_visual_decoder_subgoals.sh"
    done
    echo "launched $EVAL_NAME on GPUs 4,5,6,7"
    ;;
  status)
    tmux list-sessions 2>/dev/null | grep 'lewm-visdec-eval-' || true
    for log in "$OUTPUT_ROOT/$EVAL_NAME"/logs/*.log; do
      [[ -f "$log" ]] || continue
      echo "===== $log ====="
      tail -n 12 "$log"
    done
    ;;
  *) echo "MODE must be worker, launch, or status" >&2; exit 2 ;;
esac
