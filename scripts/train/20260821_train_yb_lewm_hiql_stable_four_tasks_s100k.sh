#!/usr/bin/env bash
set -euo pipefail

# 英博云：依次训练 LeWM 四任务的稳定化 HIQL；s100k、bs256、sg10、seed0、低学习率与弱增强。
# 官方的HIQL：原版层次化 HIQL；subgoal10、high-alpha1、low-alpha3、lr1e-4、p_aug0.2、100k。属于早期降低高层不稳定性的版本。 
CLIENT_ID=yb
DATE=$(date +%Y-%m-%d)
source /root/data/yyf/ogbench-new/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

envs=(visual-lewm-cube-single-expert-v0 visual-lewm-pusht-expert-train-v0 visual-lewm-reacher-v0 visual-lewm-tworoom-v0)
datasets=(cube_single_expert pusht_expert_train reacher tworoom)
tags=(cube pusht reacher tworoom)
gpus=(0 1 2 3)

for i in "${!envs[@]}"; do
  exp_name="${DATE}_${CLIENT_ID}_HIQLStable_lewm_${tags[$i]}_bs256_sg10_s100k_s0_aH1_aL3_e07_aug02"
  run_dir="$CLIENT_ROOT/ogbench-lewm-policy-runs/$exp_name"
  mkdir -p "$run_dir/wandb" "$run_dir/tmp"
  MUJOCO_GL=egl LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" TMPDIR="$run_dir/tmp" \
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_DIR="$run_dir/wandb" \
  "$PYTHON_BIN" main.py \
    --env_name="${envs[$i]}" --dataset_path="$LEWM_DATA_ROOT/${datasets[$i]}.lance" \
    --agent=agents/hiql.py --agent.batch_size=256 --agent.lr=1e-4 --agent.discount=0.99 \
    --agent.expectile=0.7 --agent.tau=0.005 --agent.encoder=impala_small \
    --agent.subgoal_steps=10 --agent.high_alpha=1.0 --agent.low_alpha=3.0 \
    --agent.rep_dim=10 --agent.low_actor_rep_grad=True --agent.p_aug=0.2 \
    --train_steps=100000 --seed=0 --save_dir="$run_dir" \
    --log_interval=5000 --save_interval=100000 --run_group="$exp_name" \
    --wandb_mode=offline --eval_episodes=0 --video_episodes=0
done
