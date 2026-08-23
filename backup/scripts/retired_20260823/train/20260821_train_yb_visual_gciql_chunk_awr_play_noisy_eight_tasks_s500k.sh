#!/usr/bin/env bash
set -euo pipefail

# 英博云：依次训练 Cube Single/Double/Triple 与 Scene 的 Play、Noisy 八任务 GCIQL-Chunk AWR；s500k、k5、bs512、seed0、alpha3。
CLIENT_ID=yb
DATE=$(date +%Y-%m-%d)
source /root/data/yyf/ogbench-new/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

envs=(visual-cube-single-play-v0 visual-cube-double-play-v0 visual-cube-triple-play-v0 visual-scene-play-v0 visual-cube-single-noisy-v0 visual-cube-double-noisy-v0 visual-cube-triple-noisy-v0 visual-scene-noisy-v0)
tags=(cs_play cd_play ct_play scene_play cs_noisy cd_noisy ct_noisy scene_noisy)
gpus=(0 1 2 3 4 5 6 7)

for i in "${!envs[@]}"; do
  exp_name="${DATE}_${CLIENT_ID}_GCAWR_${tags[$i]}_k5_bs512_s500k_s0_a3_e09_aug05"
  run_dir="$CLIENT_ROOT/ogbench-visual-policy-runs/$exp_name"
  mkdir -p "$run_dir/wandb" "$run_dir/tmp"
  MUJOCO_GL=egl LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" TMPDIR="$run_dir/tmp" \
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_DIR="$run_dir/wandb" \
  "$PYTHON_BIN" main.py \
    --env_name="${envs[$i]}" --agent=agents/gciql_chunk.py \
    --agent.actor_loss=awr --agent.alpha=3.0 --agent.chunk_size=5 --agent.batch_size=512 \
    --agent.lr=3e-4 --agent.discount=0.99 --agent.expectile=0.9 --agent.tau=0.005 \
    --agent.encoder=impala_small --agent.p_aug=0.5 \
    --train_steps=500000 --seed=0 --save_dir="$run_dir" \
    --log_interval=5000 --eval_interval=100000 --save_interval=100000 \
    --run_group="$exp_name" --wandb_mode=offline --eval_episodes=0 --video_episodes=0
done
