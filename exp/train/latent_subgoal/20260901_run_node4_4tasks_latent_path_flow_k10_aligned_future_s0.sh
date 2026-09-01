#!/usr/bin/env bash
set -euo pipefail

# A800 node4：GPU0–3 并行训练四任务 K5/K10 LatentPathFlow aligned-future
# 版本。唯一方法变化是 goal 候选按 action_block=5 对齐为 t+5,t+10,...；
# 其余保持原版本：frozen z192、history3、CFM、200k、bs1024、seed0、
# validation num_samples=8、EMA 0.9999、Euler-16。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

TRAIN_STEPS=${TRAIN_STEPS:-200000}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
TRAIN_SEED=${TRAIN_SEED:-0}
SUBGOAL_STEPS=10
ACTION_BLOCK=5
GOAL_SAMPLING=aligned_future
NUM_SAMPLES=8
WARMUP_STEPS=5000
VALIDATION_PAIRS=10000
EVAL_BATCH_SIZE=1024
LOG_INTERVAL=1000
EVAL_INTERVAL=10000
CHECKPOINT_INTERVAL=25000
LEWM_LATENT_ROOT=${LEWM_LATENT_ROOT:-/data-training/yyf/datasets/lewm-latents}
SUBGOAL_RUNS_ROOT=${SUBGOAL_RUNS_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10-aligned-future}

tasks=(tworoom pusht cube reacher)
gpus=(0 1 2 3)
latent_datasets=(
  "$LEWM_LATENT_ROOT/tworoom__lewm_s3072_e10_z192.h5"
  "$LEWM_LATENT_ROOT/pusht_expert_train__lewm_s666_e10_z192.h5"
  "$LEWM_LATENT_ROOT/cube_single_expert__lewm_s3072_e10_z192.h5"
  "$LEWM_LATENT_ROOT/reacher__lewm_s3072_e10_z192.h5"
)
lewm_seeds=(3072 666 3072 3072)

pids=()
for i in "${!tasks[@]}"; do
  task=${tasks[$i]}
  exp_name="latent_pathflow_${task}_lewm${lewm_seeds[$i]}_hist3_sg10_ab5_goalstride5_cfm_ns8_n${TRAIN_STEPS}_b${TRAIN_BATCH_SIZE}_s${TRAIN_SEED}"
  run_dir="$SUBGOAL_RUNS_ROOT/$exp_name"
  mkdir -p "$run_dir"

  (
    cd "$OGBENCH_ROOT/impls"
    CUDA_VISIBLE_DEVICES="${gpus[$i]}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_PLATFORMS=cuda \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" train_latent_subgoal_gcbc.py \
      --latent-dataset="${latent_datasets[$i]}" \
      --save-dir="$run_dir" \
      --exp-name="$exp_name" \
      --architecture=latent_path_flow \
      --seed="$TRAIN_SEED" \
      --split-seed=0 \
      --train-fraction=0.95 \
      --subgoal-steps="$SUBGOAL_STEPS" \
      --action-block="$ACTION_BLOCK" \
      --goal-sampling="$GOAL_SAMPLING" \
      --history-size=3 \
      --train-steps="$TRAIN_STEPS" \
      --batch-size="$TRAIN_BATCH_SIZE" \
      --hidden-dim=512 \
      --depth=4 \
      --num-heads=8 \
      --ff-dim=2048 \
      --time-dim=64 \
      --flow-sampling-steps=16 \
      --flow-solver=euler \
      --num-samples="$NUM_SAMPLES" \
      --ema-decay=0.9999 \
      --learning-rate=1e-4 \
      --final-learning-rate=1e-5 \
      --warmup-steps="$WARMUP_STEPS" \
      --weight-decay=1e-4 \
      --gradient-clip=1.0 \
      --validation-pairs="$VALIDATION_PAIRS" \
      --eval-batch-size="$EVAL_BATCH_SIZE" \
      --log-interval="$LOG_INTERVAL" \
      --eval-interval="$EVAL_INTERVAL" \
      --checkpoint-interval="$CHECKPOINT_INTERVAL" \
      --resume \
      >"$run_dir/train.log" 2>&1
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
exit "$failed"
