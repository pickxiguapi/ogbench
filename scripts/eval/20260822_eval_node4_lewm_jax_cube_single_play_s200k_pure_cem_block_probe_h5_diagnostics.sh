#!/usr/bin/env bash
set -euo pipefail

# A800 node4 GPU4：固定 LeWM-JAX 200k checkpoint，用冻结 latent 的离线线性方块位置读出器修正纯 CEM cost；Cube Single Play 每 task 1 episode，history3、H5、块内恒定、2048×5。
CLIENT_ID=node4
source /data-training/yyf/ogbench/clean-main/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

run_dir="$LEWM_JAX_RUNS_ROOT/2026-08-22_node2_LeWMJAX_cs_play_npz_impalasmall_bs128_s200k_s3072_fs5_h3_sigreg009"
output_dir="$run_dir/diagnose_node4_pure_cem_blockprobe20k_history3_h5_exec5_n2048_j5_sigma10_constant_bounded_seed42_ep1"
mkdir -p "$output_dir/videos"

CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless PYTHONUNBUFFERED=1 \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" eval_lewm_jax_ogbench.py \
  --env-name=visual-cube-single-play-v0 \
  --dataset-path="$OGBENCH_DATA_DIR/visual-cube-single-play-v0.npz" \
  --checkpoint="$run_dir/weights_step_200000.msgpack" \
  --num-eval=1 --seed=42 --planner-history-size=3 \
  --latent-probe-qpos-indices=14,15,16 --latent-probe-samples=20000 --latent-probe-ridge=0.001 \
  --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 --execution-steps=5 \
  --cem-num-samples=2048 --cem-steps=5 --cem-topk=128 --cem-var-scale=1.0 \
  --cem-temporal-parameterization=constant --cem-cost-mode=terminal \
  --video-dir="$output_dir/videos" --output="$output_dir/result.json" \
  >"$output_dir/eval.log" 2>&1
