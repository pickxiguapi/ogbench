#!/usr/bin/env bash
set -euo pipefail

# Server 11：三个 OGBench play 环境的 pi-only frozen-LeWM GCIQL-Chunk-AWR 对照。
# Q/V 各自使用可训练 IMPALA encoder；仅 actor 使用冻结 LeWM epoch10 encoder。
CLIENT_ID=11
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

POLICY_STEPS=${POLICY_STEPS:-500000}
POLICY_BATCH_SIZE=${POLICY_BATCH_SIZE:-512}
POLICY_SEED=${POLICY_SEED:-0}
P_AUG=${P_AUG:-0.0}
LEWM_EPOCH=${LEWM_EPOCH:-10}
LEWM_EPOCHS=${LEWM_EPOCHS:-10}
LEWM_BATCH_SIZE=${LEWM_BATCH_SIZE:-128}
LEWM_SEED=${LEWM_SEED:-3072}
GPU_IDS=${GPU_IDS:-"1 3 0"}
TASK_INDICES=${TASK_INDICES:-"0 1 2"}
MIN_FREE_MEMORY_MIB=${MIN_FREE_MEMORY_MIB:-60000}
WAIT_SECONDS=${WAIT_SECONDS:-60}
OGBENCH_DATA_DIR=${OGBENCH_DATA_DIR:-/data/yyf/H-LeWM/ogbench-cache/data}
LEWM_RUN_ROOT=${LEWM_RUN_ROOT:-/data/yyf/H-LeWM/ogbench-lewm-policy-runs/lewm-ogbench8}
GCIQL_RUN_ROOT=${GCIQL_RUN_ROOT:-/data/yyf/H-LeWM/ogbench-lewm-policy-runs/gciql-chunk-ogbench8}

envs=(visual-cube-single-play-v0 visual-cube-double-play-v0 visual-cube-triple-play-v0)
tags=(cs_play cd_play ct_play)
read -r -a gpu_ids <<<"$GPU_IDS"
read -r -a task_indices <<<"$TASK_INDICES"

if (( ${#gpu_ids[@]} != ${#task_indices[@]} )); then
  echo "GPU_IDS and TASK_INDICES must have the same length" >&2
  exit 2
fi
for task_index in "${task_indices[@]}"; do
  if (( task_index < 0 || task_index >= ${#envs[@]} )); then
    echo "Invalid task index: $task_index" >&2
    exit 2
  fi
done

lewm_checkpoints=()
for i in "${!envs[@]}"; do
  checkpoint="$LEWM_RUN_ROOT/lewm_ogbench8_${tags[$i]}_e${LEWM_EPOCHS}_bs${LEWM_BATCH_SIZE}_s${LEWM_SEED}/weights_epoch_${LEWM_EPOCH}.msgpack"
  dataset="$OGBENCH_DATA_DIR/${envs[$i]}.npz"
  if [[ ! -s "$checkpoint" ]]; then
    echo "Frozen LeWM checkpoint not found: $checkpoint" >&2
    exit 2
  fi
  if [[ ! -s "$dataset" ]]; then
    echo "OGBench dataset not found: $dataset" >&2
    exit 2
  fi
  lewm_checkpoints+=("$checkpoint")
done

wait_for_gpu_memory() {
  local gpu=$1
  local free_memory
  while true; do
    free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
    if [[ -n "$free_memory" ]] && (( free_memory >= MIN_FREE_MEMORY_MIB )); then
      return
    fi
    echo "[$(date '+%F %T %Z')] GPU $gpu has ${free_memory:-unknown} MiB free; waiting for ${MIN_FREE_MEMORY_MIB} MiB"
    sleep "$WAIT_SECONDS"
  done
}

train_task() {
  local gpu=$1
  local task_index=$2
  local tag=${tags[$task_index]}
  local exp_name="gc8_${tag}_pi_n${POLICY_STEPS}_b${POLICY_BATCH_SIZE}_a${P_AUG}_sd${POLICY_SEED}"
  local run_dir="$GCIQL_RUN_ROOT/$exp_name"

  if [[ -s "$run_dir/params_${POLICY_STEPS}.pkl" ]]; then
    echo "[$(date '+%F %T %Z')] Skip completed $exp_name"
    return
  fi
  wait_for_gpu_memory "$gpu"
  mkdir -p "$run_dir"
  echo "[$(date '+%F %T %Z')] Start GPU $gpu $exp_name"
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" "$OGBENCH_ROOT/impls/train_gciql_chunk.py" \
    --env_name="${envs[$task_index]}" \
    --dataset_path="$OGBENCH_DATA_DIR/${envs[$task_index]}.npz" \
    --save_dir="$run_dir" \
    --representation_mode=pi \
    --lewm_checkpoint="${lewm_checkpoints[$task_index]}" \
    --actor_loss=awr --alpha=3.0 --pixel_encoder=impala_small --p_aug="$P_AUG" \
    --train_steps="$POLICY_STEPS" --batch_size="$POLICY_BATCH_SIZE" --seed="$POLICY_SEED" \
    --chunk_size=5 --lr=3e-4 --discount=0.99 --expectile=0.9 --tau=0.005 \
    --log_interval=5000 --save_interval=100000 \
    >"$run_dir/train.log" 2>&1
  echo "[$(date '+%F %T %Z')] Finish GPU $gpu $exp_name"
}

pids=()
for i in "${!task_indices[@]}"; do
  train_task "${gpu_ids[$i]}" "${task_indices[$i]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
