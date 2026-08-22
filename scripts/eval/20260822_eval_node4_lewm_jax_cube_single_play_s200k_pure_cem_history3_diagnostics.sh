#!/usr/bin/env bash
set -euo pipefail

# A800 node4 GPU4：诊断固定 LeWM-JAX 200k checkpoint 的纯 bounded CEM 控制行为；Cube Single Play 每 task 1 episode，history3、H1、1024×3，并保存动作/物体位移统计与视频。
CLIENT_ID=node4
source /data-training/yyf/ogbench/clean-main/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

run_dir="$LEWM_JAX_RUNS_ROOT/2026-08-22_node2_LeWMJAX_cs_play_npz_impalasmall_bs128_s200k_s3072_fs5_h3_sigreg009"
output_dir="$run_dir/diagnose_node4_pure_cem_history3_h1_exec5_n1024_j3_sigma05_bounded_seed42_ep1"
mkdir -p "$output_dir/videos"

CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless PYTHONUNBUFFERED=1 \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" eval_lewm_jax_ogbench.py \
  --env-name=visual-cube-single-play-v0 \
  --dataset-path="$OGBENCH_DATA_DIR/visual-cube-single-play-v0.npz" \
  --checkpoint="$run_dir/weights_step_200000.msgpack" \
  --num-eval=1 --seed=42 --planner-history-size=3 \
  --cem-horizon=1 --cem-receding-horizon=1 --action-block=5 --execution-steps=5 \
  --cem-num-samples=1024 --cem-steps=3 --cem-topk=128 --cem-var-scale=0.5 \
  --cem-cost-mode=terminal --video-dir="$output_dir/videos" \
  --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1
