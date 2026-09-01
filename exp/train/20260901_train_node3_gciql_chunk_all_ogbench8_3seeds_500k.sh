#!/usr/bin/env bash
set -euo pipefail

# A800 node3：OGBench-Env-8Tasks shared-all frozen-LeWM GCIQL-Chunk-AWR，3 training seeds、500k、bs512、k5、alpha3、无增强。
CLIENT_ID=node3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

POLICY_STEPS=${POLICY_STEPS:-500000}
POLICY_BATCH_SIZE=${POLICY_BATCH_SIZE:-512}
POLICY_SEEDS=${POLICY_SEEDS:-"0 1 2"}
P_AUG=${P_AUG:-0.0}
LEWM_EPOCH=${LEWM_EPOCH:-10}
LEWM_EPOCHS=${LEWM_EPOCHS:-10}
LEWM_BATCH_SIZE=${LEWM_BATCH_SIZE:-128}
LEWM_SEED=${LEWM_SEED:-3072}
LEWM_RUN_ROOT=${LEWM_RUN_ROOT:-$CLIENT_ROOT/ogbench-lewm-policy-runs/lewm-ogbench8}
GCIQL_RUN_ROOT=${GCIQL_RUN_ROOT:-$CLIENT_ROOT/ogbench-lewm-policy-runs/gciql-chunk-ogbench8}

read -r -a policy_seeds <<<"$POLICY_SEEDS"
if (( ${#policy_seeds[@]} != 3 )); then
  echo "POLICY_SEEDS must contain exactly three seeds; got: $POLICY_SEEDS" >&2
  exit 2
fi

envs=(visual-cube-single-play-v0 visual-cube-double-play-v0 visual-cube-triple-play-v0 visual-scene-play-v0 visual-cube-single-noisy-v0 visual-cube-double-noisy-v0 visual-cube-triple-noisy-v0 visual-scene-noisy-v0)
tags=(cs_play cd_play ct_play scene_play cs_noisy cd_noisy ct_noisy scene_noisy)
lewm_checkpoints=()

for tag in "${tags[@]}"; do
  checkpoint="$LEWM_RUN_ROOT/lewm_ogbench8_${tag}_e${LEWM_EPOCHS}_bs${LEWM_BATCH_SIZE}_s${LEWM_SEED}/weights_epoch_${LEWM_EPOCH}.msgpack"
  if [[ ! -s "$checkpoint" ]]; then
    echo "Frozen LeWM checkpoint not found: $checkpoint" >&2
    exit 2
  fi
  lewm_checkpoints+=("$checkpoint")
done

cd "$OGBENCH_ROOT/impls"

wait_for_gpu() {
  local gpu=$1
  local gpu_uuid
  gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n "$((gpu + 1))p")
  while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
    | grep -Fxq "$gpu_uuid"; do
    echo "[$(date '+%F %T %Z')] GPU $gpu is occupied; waiting"
    sleep 60
  done
}

train_task() {
  local gpu=$1
  local task_index=$2
  local seed=$3
  local tag=${tags[$task_index]}
  local exp_name="gc8_${tag}_all_n${POLICY_STEPS}_b${POLICY_BATCH_SIZE}_a${P_AUG}_sd${seed}"
  local run_dir="$GCIQL_RUN_ROOT/$exp_name"

  if [[ -s "$run_dir/params_${POLICY_STEPS}.pkl" ]]; then
    echo "[$(date '+%F %T %Z')] Skip completed $exp_name"
    return
  fi
  mkdir -p "$run_dir"
  echo "[$(date '+%F %T %Z')] Start GPU $gpu $exp_name"
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_gciql_chunk.py \
    --env_name="${envs[$task_index]}" \
    --save_dir="$run_dir" \
    --representation_mode=all \
    --lewm_checkpoint="${lewm_checkpoints[$task_index]}" \
    --actor_loss=awr --alpha=3.0 --pixel_encoder=impala_small --p_aug="$P_AUG" \
    --train_steps="$POLICY_STEPS" --batch_size="$POLICY_BATCH_SIZE" --seed="$seed" \
    --chunk_size=5 --lr=3e-4 --discount=0.99 --expectile=0.9 --tau=0.005 \
    --log_interval=5000 --save_interval=100000 \
    >"$run_dir/train.log" 2>&1
  echo "[$(date '+%F %T %Z')] Finish GPU $gpu $exp_name"
}

run_worker() {
  local gpu=$1
  shift
  local task_indices=("$@")
  for task_index in "${task_indices[@]}"; do
    for seed in "${policy_seeds[@]}"; do
      wait_for_gpu "$gpu"
      train_task "$gpu" "$task_index" "$seed"
    done
  done
}

pids=()
run_worker 1 0 7 & pids+=("$!")
run_worker 2 1 & pids+=("$!")
run_worker 3 2 & pids+=("$!")
run_worker 4 3 & pids+=("$!")
run_worker 5 4 & pids+=("$!")
run_worker 6 5 & pids+=("$!")
run_worker 7 6 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
