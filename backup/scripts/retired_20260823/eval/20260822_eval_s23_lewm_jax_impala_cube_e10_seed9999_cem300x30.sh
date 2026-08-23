#!/usr/bin/env bash
set -euo pipefail

# Server 23: LeWM-JAX IMPALA seed9999 Cube epoch-10; dataset-goal evaluation, 50 episodes.
CLIENT_ID=23
source /home/dzb/ogbench/scripts/client_env.sh

exp_name=LeWMJAX_impala_lance_cube_single_bs128_e10_seed9999_fs5_h3_sigreg009_jpeg95
run_dir="/data/dzb/stablewm-data/lewm-jax-runs/$exp_name"
output_dir="$run_dir/eval_cem300x30_seed42_20260822"

test -s "$run_dir/weights_epoch_10.msgpack"
mkdir -p "$output_dir/videos"
cd "$OGBENCH_ROOT/impls"

CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" eval_lewm_jax_cem.py \
  --task=cube --checkpoint="$run_dir/weights_epoch_10.msgpack" \
  --data-root="$LEWM_DATA_ROOT" \
  --num-eval=50 --seed=42 --goal-offset-steps=25 --eval-budget=50 \
  --cem-horizon=5 --cem-receding-horizon=5 --action-block=5 \
  --cem-num-samples=300 --cem-steps=30 --cem-topk=30 --cem-var-scale=1.0 \
  --video-dir="$output_dir/videos" --output="$output_dir/cube.json" \
  2>&1 | tee "$output_dir/eval.log"
