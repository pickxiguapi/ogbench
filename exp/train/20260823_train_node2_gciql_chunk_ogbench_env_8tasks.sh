#!/usr/bin/env bash
set -euo pipefail

# A800 node2：八卡并行训练 OGBench-Env-8Tasks GCIQL-Chunk；REPRESENTATION_MODE 可设 independent/pi/qv/all，默认关闭增强以做受控消融。
CLIENT_ID=node2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
REPRESENTATION_MODE=${REPRESENTATION_MODE:-independent}
P_AUG=${P_AUG:-0.0}
POLICY_STEPS=${POLICY_STEPS:-500000}
POLICY_BATCH_SIZE=${POLICY_BATCH_SIZE:-512}
POLICY_SEED=${POLICY_SEED:-0}
LEWM_STEPS=${LEWM_STEPS:-200000}
LEWM_SEED=${LEWM_SEED:-3072}
LEWM_BATCH_SIZE=${LEWM_BATCH_SIZE:-128}
source "$OGBENCH_ROOT/scripts/client_env.sh"
cd "$OGBENCH_ROOT/impls"

case "$REPRESENTATION_MODE" in independent|pi|qv|all) ;; *) echo "REPRESENTATION_MODE must be independent, pi, qv, or all" >&2; exit 2 ;; esac
case "$REPRESENTATION_MODE" in independent) MODE_TAG=ind ;; *) MODE_TAG=$REPRESENTATION_MODE ;; esac
envs=(visual-cube-single-play-v0 visual-cube-double-play-v0 visual-cube-triple-play-v0 visual-scene-play-v0 visual-cube-single-noisy-v0 visual-cube-double-noisy-v0 visual-cube-triple-noisy-v0 visual-scene-noisy-v0)
tags=(cs_play cd_play ct_play scene_play cs_noisy cd_noisy ct_noisy scene_noisy)
pids=()

for i in "${!envs[@]}"; do
  exp_name="gc8_${tags[$i]}_${MODE_TAG}_n${POLICY_STEPS}_b${POLICY_BATCH_SIZE}_a${P_AUG}_sd${POLICY_SEED}"
  if (( ${#exp_name} >= 64 )); then
    echo "Experiment name must be shorter than 64 characters: $exp_name" >&2
    exit 2
  fi
  run_dir="$CLIENT_ROOT/lewm-final/gciql-chunk-ogbench8/$exp_name"
  lewm_args=()
  if [[ "$REPRESENTATION_MODE" != independent ]]; then
    lewm_dir="$CLIENT_ROOT/lewm-final/lewm-ogbench8/lewm_ogbench8_${tags[$i]}_s${LEWM_STEPS}_bs${LEWM_BATCH_SIZE}_s${LEWM_SEED}"
    lewm_args=(--lewm_checkpoint="$lewm_dir/weights_step_${LEWM_STEPS}.msgpack")
  fi
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES=$i XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_gciql_chunk.py \
    --env_name="${envs[$i]}" --save_dir="$run_dir" \
    --representation_mode="$REPRESENTATION_MODE" "${lewm_args[@]}" \
    --actor_loss=awr --alpha=3.0 --pixel_encoder=impala_small --p_aug="$P_AUG" \
    --train_steps="$POLICY_STEPS" --batch_size="$POLICY_BATCH_SIZE" --seed="$POLICY_SEED" \
    --chunk_size=5 --lr=3e-4 --discount=0.99 --expectile=0.9 --tau=0.005 \
    --log_interval=5000 --save_interval="$POLICY_STEPS" >"$run_dir/train.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
