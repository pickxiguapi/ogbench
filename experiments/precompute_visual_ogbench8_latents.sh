#!/usr/bin/env bash
set -euo pipefail

# Fill in these paths before running this script.
OGBENCH_DATA_ROOT=""
EXPERIMENT_ROOT="outputs"
VISUAL_OGBENCH_EXPERIMENT_ROOT="$EXPERIMENT_ROOT/visual-ogbench8"
ENVS=(
  visual-cube-single-play-v0
  visual-cube-double-play-v0
  visual-cube-triple-play-v0
  visual-scene-play-v0
  visual-cube-single-noisy-v0
  visual-cube-double-noisy-v0
  visual-cube-triple-noisy-v0
  visual-scene-noisy-v0
)
TAGS=(cs_play cd_play ct_play scene_play cs_noisy cd_noisy ct_noisy scene_noisy)
GPU_IDS=(0 1 2 3 4 5 6 7)

encode_one() {
  local gpu=$1 index=$2 env_name=${ENVS[$2]} tag=${TAGS[$2]}
  local output="$VISUAL_OGBENCH_EXPERIMENT_ROOT/latents/$tag.h5"
  mkdir -p "$VISUAL_OGBENCH_EXPERIMENT_ROOT/latents"
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python impls/precompute_lewm_npz_latents.py \
      --env-name="$env_name" --npz-path="$OGBENCH_DATA_ROOT/$env_name.npz" \
      --checkpoint="$VISUAL_OGBENCH_EXPERIMENT_ROOT/lewm/$tag/weights_epoch_10.msgpack" \
      --output="$output" --batch-size=512 --output-dtype=float32 \
      --flush-every-batches=20 --log-every-batches=20 \
      2>&1 | tee "$VISUAL_OGBENCH_EXPERIMENT_ROOT/latents/$tag.log"
}

pids=()
for index in "${!ENVS[@]}"; do
  encode_one "${GPU_IDS[$index]}" "$index" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
