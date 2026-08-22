#!/usr/bin/env bash
set -euo pipefail

# A800 node4 GPU4：固定 LeWM-JAX 200k checkpoint，联合当前/目标图像检索真实H5计划并完整执行25步，避免抓取阶段被频繁重规划打断；Cube Single Play 每 task 1 episode，无 policy。
CLIENT_ID=node4
source /data-training/yyf/ogbench/clean-main/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

run_dir="$LEWM_JAX_RUNS_ROOT/2026-08-22_node2_LeWMJAX_cs_play_npz_impalasmall_bs128_s200k_s3072_fs5_h3_sigreg009"
output_dir="$run_dir/diagnose_node4_pure_wm_goalknn131072_off100_rank100_blockprobe20k_h5_exec25_n512_seed42_ep1"
mkdir -p "$output_dir/videos"

CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless PYTHONUNBUFFERED=1 \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" eval_lewm_jax_ogbench.py \
  --env-name=visual-cube-single-play-v0 \
  --dataset-path="$OGBENCH_DATA_DIR/visual-cube-single-play-v0.npz" \
  --checkpoint="$run_dir/weights_step_200000.msgpack" \
  --num-eval=1 --seed=42 --planner-history-size=1 \
  --latent-probe-qpos-indices=14,15,16 --latent-probe-samples=20000 --latent-probe-ridge=0.001 \
  --cem-horizon=5 --cem-receding-horizon=5 --action-block=5 --execution-steps=25 \
  --cem-num-samples=512 --cem-steps=1 --cem-topk=64 --cem-var-scale=1.0 \
  --cem-empirical-action-reservoir-size=131072 --cem-empirical-full-plans \
  --cem-empirical-state-conditioned --cem-empirical-goal-offset-steps=100 \
  --cem-empirical-goal-distance-weight=1 --cem-empirical-context-rank-penalty=100 \
  --cem-return-best-candidate --cem-cost-mode=terminal \
  --video-dir="$output_dir/videos" --output="$output_dir/result.json" \
  >"$output_dir/eval.log" 2>&1
