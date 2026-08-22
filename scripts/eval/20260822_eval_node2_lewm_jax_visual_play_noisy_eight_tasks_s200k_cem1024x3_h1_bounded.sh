#!/usr/bin/env bash
set -euo pipefail

# A800 node2：纯 LeWM-JAX + bounded CEM 评测 Visual Play/Noisy 八环境；固定 200k checkpoint，每个内部 task 20 episodes，H1、1024×3、sigma0.5、action-block5。
CLIENT_ID=node2
DATE=2026-08-22
source /home/yyf/ogbench-main/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

envs=(visual-cube-single-play-v0 visual-cube-double-play-v0 visual-cube-triple-play-v0 visual-scene-play-v0 visual-cube-single-noisy-v0 visual-cube-double-noisy-v0 visual-cube-triple-noisy-v0 visual-scene-noisy-v0)
tags=(cs_play cd_play ct_play scene_play cs_noisy cd_noisy ct_noisy scene_noisy)
gpus=(0 1 2 3 4 5 6 7)
pids=()

for i in "${!envs[@]}"; do
  exp_name="${DATE}_${CLIENT_ID}_LeWMJAX_${tags[$i]}_npz_impalasmall_bs128_s200k_s3072_fs5_h3_sigreg009"
  run_dir="$LEWM_JAX_RUNS_ROOT/$exp_name"
  output_dir="$run_dir/eval_pure_cem1024x3_h1_rh1_ab5_sigma05_bounded_seed42_ep20_s200k"
  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless PYTHONUNBUFFERED=1 \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" eval_lewm_jax_ogbench.py \
    --env-name="${envs[$i]}" --dataset-path="$OGBENCH_DATA_DIR/${envs[$i]}.npz" \
    --checkpoint="$run_dir/weights_step_200000.msgpack" \
    --num-eval=20 --seed=42 \
    --cem-horizon=1 --cem-receding-horizon=1 --action-block=5 --execution-steps=5 \
    --cem-num-samples=1024 --cem-steps=3 --cem-topk=128 --cem-var-scale=0.5 \
    --cem-cost-mode=terminal \
    --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "launched gpu=${gpus[$i]} env=${envs[$i]} pid=$pid log=$output_dir/eval.log"
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
