#!/usr/bin/env bash
set -euo pipefail

# A800 node4：训练 H50 LatentPathFlow predictor 消融。四任务分别使用绑定的
# frozen LeWM z192 cache，比较参数量约 18.7M 的 history3 MLP、单 K10
# EndpointFlow 与 K5/K10 LatentPathFlow；三者共享 full-offset uniform-future
# 数据、固定 H50 validation manifest、200k updates、bs1024、EMA 与 seeds
# 0/1/42。每批最多使用 8 张 GPU 并行，所有正式训练均从本 Bash 发起。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
ARCHITECTURES=${ARCHITECTURES:-"history_mlp endpoint_flow latent_path_flow"}
TRAIN_SEEDS=${TRAIN_SEEDS:-"0 1 42"}
TRAIN_STEPS=${TRAIN_STEPS:-200000}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
VALIDATION_PAIRS=${VALIDATION_PAIRS:-10000}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1024}
WARMUP_STEPS=${WARMUP_STEPS:-5000}
LOG_INTERVAL=${LOG_INTERVAL:-1000}
EVAL_INTERVAL=${EVAL_INTERVAL:-10000}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-25000}
RUNS_ROOT=${RUNS_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-predictor-h50-ablation}
MANIFEST_ROOT=${MANIFEST_ROOT:-$RUNS_ROOT/manifests}

source "$OGBENCH_ROOT/scripts/client_env.sh"

tasks=(cube pusht reacher tworoom)
lewm_seeds=(3072 666 3072 3072)
latent_datasets=(
  /data-training/yyf/datasets/lewm-latents/cube_single_expert__lewm_s3072_e10_z192.h5
  /data-training/yyf/datasets/lewm-latents/pusht_expert_train__lewm_s666_e10_z192.h5
  /data-training/yyf/datasets/lewm-latents/reacher__lewm_s3072_e10_z192.h5
  /data-training/yyf/datasets/lewm-latents/tworoom__lewm_s3072_e10_z192.h5
)

read -r -a gpu_ids <<< "$GPU_IDS"
read -r -a architectures <<< "$ARCHITECTURES"
read -r -a train_seeds <<< "$TRAIN_SEEDS"
if (( ${#gpu_ids[@]} == 0 || ${#gpu_ids[@]} > 8 )); then
  echo "GPU_IDS must contain between one and eight GPU IDs." >&2
  exit 2
fi
gpu_count=${#gpu_ids[@]}

mkdir -p "$MANIFEST_ROOT"
for i in "${!tasks[@]}"; do
  (
    cd "$OGBENCH_ROOT/impls"
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" create_latent_subgoal_validation_manifest.py \
      --latent-dataset="${latent_datasets[$i]}" \
      --output="$MANIFEST_ROOT/${tasks[$i]}_h50_hist3_k10_n${VALIDATION_PAIRS}.npz" \
      --split-seed=0 --train-fraction=0.95 \
      --num-pairs="$VALIDATION_PAIRS" --history-size=3 \
      --goal-offset=50 --subgoal-steps=10 --action-block=5 --seed=1
  )
done

variant_tasks=()
variant_task_indices=()
variant_architectures=()
variant_seeds=()
for architecture in "${architectures[@]}"; do
  if [[ "$architecture" != history_mlp && "$architecture" != endpoint_flow && "$architecture" != latent_path_flow ]]; then
    echo "Unknown architecture: $architecture" >&2
    exit 2
  fi
  for train_seed in "${train_seeds[@]}"; do
    for i in "${!tasks[@]}"; do
      variant_tasks+=("${tasks[$i]}")
      variant_task_indices+=("$i")
      variant_architectures+=("$architecture")
      variant_seeds+=("$train_seed")
    done
  done
done

run_training() {
  local gpu_id=$1
  local task=$2
  local task_index=$3
  local architecture=$4
  local train_seed=$5
  local architecture_tag=$architecture
  local exp_name="h50_${architecture_tag}_${task}_lewm${lewm_seeds[$task_index]}_hist3_k10_pmatch18m_n${TRAIN_STEPS}_b${TRAIN_BATCH_SIZE}_s${train_seed}"
  local run_dir="$RUNS_ROOT/$exp_name"
  local manifest="$MANIFEST_ROOT/${task}_h50_hist3_k10_n${VALIDATION_PAIRS}.npz"
  mkdir -p "$run_dir"

  if [[ "$architecture" == history_mlp ]]; then
    (
      cd "$OGBENCH_ROOT/impls"
      CUDA_VISIBLE_DEVICES="$gpu_id" XLA_PYTHON_CLIENT_PREALLOCATE=false \
      JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 \
      PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
      "$PYTHON_BIN" train_latent_subgoal_gcbc.py \
        --latent-dataset="${latent_datasets[$task_index]}" \
        --validation-manifest="$manifest" \
        --save-dir="$run_dir" --exp-name="$exp_name" \
        --architecture=history_mlp --hidden-dims 2048 2048 2048 2048 2048 \
        --history-size=3 --seed="$train_seed" --split-seed=0 --train-fraction=0.95 \
        --subgoal-steps=10 --action-block=5 --goal-sampling=uniform_future \
        --train-steps="$TRAIN_STEPS" --batch-size="$TRAIN_BATCH_SIZE" \
        --ema-decay=0.9999 --learning-rate=1e-4 --final-learning-rate=1e-5 \
        --warmup-steps="$WARMUP_STEPS" --weight-decay=1e-4 --gradient-clip=1.0 \
        --validation-pairs="$VALIDATION_PAIRS" --eval-batch-size="$EVAL_BATCH_SIZE" \
        --log-interval="$LOG_INTERVAL" --eval-interval="$EVAL_INTERVAL" \
        --checkpoint-interval="$CHECKPOINT_INTERVAL" \
        --resume 2>&1 | tee -a "$run_dir/train.log"
    )
  else
    (
      cd "$OGBENCH_ROOT/impls"
      CUDA_VISIBLE_DEVICES="$gpu_id" XLA_PYTHON_CLIENT_PREALLOCATE=false \
      JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 \
      PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
      "$PYTHON_BIN" train_latent_subgoal_gcbc.py \
        --latent-dataset="${latent_datasets[$task_index]}" \
        --validation-manifest="$manifest" \
        --save-dir="$run_dir" --exp-name="$exp_name" \
        --architecture="$architecture" \
        --history-size=3 --hidden-dim=512 --depth=4 --num-heads=8 --ff-dim=2048 --time-dim=64 \
        --seed="$train_seed" --split-seed=0 --train-fraction=0.95 \
        --subgoal-steps=10 --action-block=5 --goal-sampling=uniform_future \
        --train-steps="$TRAIN_STEPS" --batch-size="$TRAIN_BATCH_SIZE" \
        --flow-sampling-steps=16 --flow-solver=euler --num-samples=1 \
        --ema-decay=0.9999 --learning-rate=1e-4 --final-learning-rate=1e-5 \
        --warmup-steps="$WARMUP_STEPS" --weight-decay=1e-4 --gradient-clip=1.0 \
        --validation-pairs="$VALIDATION_PAIRS" --eval-batch-size="$EVAL_BATCH_SIZE" \
        --log-interval="$LOG_INTERVAL" --eval-interval="$EVAL_INTERVAL" \
        --checkpoint-interval="$CHECKPOINT_INTERVAL" \
        --resume 2>&1 | tee -a "$run_dir/train.log"
    )
  fi
}

failed=0
for (( base=0; base<${#variant_tasks[@]}; base+=gpu_count )); do
  batch_pids=()
  for (( slot=0; slot<gpu_count && base+slot<${#variant_tasks[@]}; slot++ )); do
    index=$((base + slot))
    run_training "${gpu_ids[$slot]}" \
      "${variant_tasks[$index]}" \
      "${variant_task_indices[$index]}" \
      "${variant_architectures[$index]}" \
      "${variant_seeds[$index]}" &
    batch_pids+=("$!")
  done
  for pid in "${batch_pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
done
exit "$failed"
