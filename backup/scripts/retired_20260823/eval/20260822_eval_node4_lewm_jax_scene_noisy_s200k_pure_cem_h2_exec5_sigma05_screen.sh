#!/usr/bin/env bash
set -euo pipefail

# A800 node4 GPU4：固定 LeWM-JAX 200k checkpoint，在旧评测唯一有成功的 Scene Noisy 复核纯 bounded CEM；5 episodes/task，H2、执行5步、512×3、sigma0.5。
CLIENT_ID=node4
source /data-training/yyf/ogbench/clean-main/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

run_dir="$LEWM_JAX_RUNS_ROOT/2026-08-22_node2_LeWMJAX_scene_noisy_npz_impalasmall_bs128_s200k_s3072_fs5_h3_sigreg009"
output_dir="$run_dir/screen_node4_pure_cem_h2_exec5_n512_j3_sigma05_bounded_seed42_ep5"
mkdir -p "$output_dir"

CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless PYTHONUNBUFFERED=1 \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" eval_lewm_jax_ogbench.py \
  --env-name=visual-scene-noisy-v0 \
  --dataset-path="$OGBENCH_DATA_DIR/visual-scene-noisy-v0.npz" \
  --checkpoint="$run_dir/weights_step_200000.msgpack" \
  --num-eval=5 --seed=42 \
  --cem-horizon=2 --cem-receding-horizon=1 --action-block=5 --execution-steps=5 \
  --cem-num-samples=512 --cem-steps=3 --cem-topk=64 --cem-var-scale=0.5 \
  --cem-cost-mode=terminal \
  --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1
