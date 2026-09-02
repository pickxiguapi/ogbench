#!/usr/bin/env bash
set -euo pipefail

# Server 11: queue the four OGBench noisy environments after the four play
# environments, using fully independent GCIQL-Chunk-AWR pixel encoders.
CLIENT_ID=11
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

POLICY_STEPS=${POLICY_STEPS:-500000}
POLICY_BATCH_SIZE=${POLICY_BATCH_SIZE:-512}
POLICY_SEED=${POLICY_SEED:-0}
P_AUG=${P_AUG:-0.0}
GPU_IDS=${GPU_IDS:-"0 1 2 3"}
MIN_FREE_MEMORY_MIB=${MIN_FREE_MEMORY_MIB:-60000}
WAIT_SECONDS=${WAIT_SECONDS:-30}
OGBENCH_DATA_DIR=${OGBENCH_DATA_DIR:-/data/yyf/H-LeWM/ogbench-cache/data}
GCIQL_RUN_ROOT=${GCIQL_RUN_ROOT:-/data/yyf/H-LeWM/ogbench-lewm-policy-runs/gciql-chunk-ogbench8}

envs=(
  visual-cube-single-noisy-v0
  visual-cube-double-noisy-v0
  visual-cube-triple-noisy-v0
  visual-scene-noisy-v0
)
tags=(cs_noisy cd_noisy ct_noisy scene_noisy)
read -r -a gpu_ids <<<"$GPU_IDS"

if (( ${#gpu_ids[@]} != ${#envs[@]} )); then
  echo "GPU_IDS must contain exactly four whitespace-separated GPU IDs." >&2
  exit 2
fi

wait_for_dataset_pair() {
  local dataset=$1
  local validation_dataset=${dataset%.npz}-val.npz
  while [[ ! -s "$dataset" || ! -s "$validation_dataset" ]]; do
    echo "[$(date '+%F %T %Z')] Waiting for dataset pair: $dataset"
    sleep "$WAIT_SECONDS"
  done
}

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
  local dataset="$OGBENCH_DATA_DIR/${envs[$task_index]}.npz"
  local exp_name="gc8_${tag}_ind_n${POLICY_STEPS}_b${POLICY_BATCH_SIZE}_a${P_AUG}_sd${POLICY_SEED}"
  local run_dir="$GCIQL_RUN_ROOT/$exp_name"

  if [[ -s "$run_dir/params_${POLICY_STEPS}.pkl" ]]; then
    echo "[$(date '+%F %T %Z')] Skip completed $exp_name"
    return
  fi
  wait_for_dataset_pair "$dataset"
  wait_for_gpu_memory "$gpu"
  mkdir -p "$run_dir"
  echo "[$(date '+%F %T %Z')] Start GPU $gpu $exp_name"
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" "$OGBENCH_ROOT/impls/train_gciql_chunk.py" \
    --env_name="${envs[$task_index]}" \
    --dataset_path="$dataset" \
    --save_dir="$run_dir" \
    --representation_mode=independent \
    --actor_loss=awr --alpha=3.0 --pixel_encoder=impala_small --p_aug="$P_AUG" \
    --train_steps="$POLICY_STEPS" --batch_size="$POLICY_BATCH_SIZE" --seed="$POLICY_SEED" \
    --chunk_size=5 --lr=3e-4 --discount=0.99 --expectile=0.9 --tau=0.005 \
    --log_interval=5000 --save_interval=100000 \
    >"$run_dir/train.log" 2>&1
  echo "[$(date '+%F %T %Z')] Finish GPU $gpu $exp_name"
}

pids=()
for i in "${!envs[@]}"; do
  train_task "${gpu_ids[$i]}" "$i" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
