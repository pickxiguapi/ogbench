#!/usr/bin/env bash
set -euo pipefail

# Server 23：评测 LeWM-JAX Visual Cube single/double epoch-10；OGBench 五任务各 50 episodes，CEM 300×30。
task="${1:-}"
CLIENT_ID=23
source /home/dzb/ogbench/scripts/client_env.sh

case "$task" in
  single) gpu=2 ;;
  double) gpu=3 ;;
  *)
    echo "Usage: bash scripts/eval/20260819_eval_s23_lewm_jax_visual_cube_play_e10_cem300x30.sh {single|double}" >&2
    exit 2
    ;;
esac

env_name="visual-cube-${task}-play-v0"
exp_name="LeWMJAX_ogbench_visual_cube_${task}_play_impalasmall_bs128_e10_seed3072_fs1_h3_bf16_s23"
run_dir="/data/dzb/stablewm-data/lewm-jax-runs/$exp_name"
output_dir="$run_dir/eval_ogbench_cem300x30_seed42_e10"
mkdir -p "$output_dir"
cd "$OGBENCH_ROOT/impls"

CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" eval_lewm_jax_ogbench.py \
  --env-name="$env_name" --dataset-path="/home/dzb/.ogbench/data/$env_name.npz" \
  --checkpoint="$run_dir/weights_epoch_10.msgpack" \
  --num-eval=50 --seed=42 \
  --cem-horizon=5 --cem-receding-horizon=5 \
  --cem-num-samples=300 --cem-steps=30 --cem-topk=30 \
  --output="$output_dir/$task.json" 2>&1 | tee "$output_dir/eval.log"
