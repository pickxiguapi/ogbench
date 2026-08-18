#!/usr/bin/env bash
set -euo pipefail

# 英博云：nohup 后台并行训练四个 Play 和四个 Noise 视觉任务的 GCIQL-Chunk AWR；8 卡各一任务，s100k、k5、bs512、seed0。
CLIENT_ID=yb
DATE=$(date +%Y-%m-%d)
source /root/data/yyf/ogbench-new/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

envs=(visual-cube-single-play-v0 visual-cube-double-play-v0 visual-cube-triple-play-v0 visual-scene-play-v0 visual-cube-single-noise-v0 visual-cube-double-noise-v0 visual-cube-triple-noise-v0 visual-scene-noise-v0)
tags=(cube_single_play cube_double_play cube_triple_play scene_play cube_single_noise cube_double_noise cube_triple_noise scene_noise)
gpus=(0 1 2 3 4 5 6 7)

for i in "${!envs[@]}"; do
  exp_name="${DATE}_${CLIENT_ID}_GCIQLChunkAWR_${tags[$i]}_k5_bs512_s100k_seed0_alpha3_exp09_aug05"
  run_dir="$CLIENT_ROOT/ogbench-gciql-chunk-awr-runs/$exp_name"
  mkdir -p "$run_dir/wandb" "$run_dir/tmp"
  MUJOCO_GL=egl LD_LIBRARY_PATH="$EGL_LIB_DIR" TMPDIR="$run_dir/tmp" \
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_DIR="$run_dir/wandb" \
  nohup "$PYTHON_BIN" main.py \
    --env_name="${envs[$i]}" --agent=agents/gciql_chunk.py \
    --agent.actor_loss=awr --agent.alpha=3.0 --agent.chunk_size=5 --agent.batch_size=512 \
    --agent.lr=3e-4 --agent.discount=0.99 --agent.expectile=0.9 --agent.tau=0.005 \
    --agent.encoder=impala_small --agent.p_aug=0.5 \
    --train_steps=100000 --seed=0 --save_dir="$run_dir" \
    --log_interval=5000 --eval_interval=100000 --save_interval=100000 \
    --run_group="$exp_name" --wandb_mode=offline --eval_episodes=0 --video_episodes=0 \
    > "$run_dir/train.log" 2>&1 &
done
