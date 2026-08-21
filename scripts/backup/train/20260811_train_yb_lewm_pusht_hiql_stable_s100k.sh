#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/root/data/yyf/ogbench"
DATA_ROOT="/root/data/yyf/stable-worldmodel/datasets"
RUNS_ROOT="/root/data/yyf/lewm-runs"
GPU_ID=6
EXP_NAME="HIQL_ogbench_pusht_bs256_s100k_seed0_sg10_ha1_la3_lr1e4_rep10_aug02_repgrad"
LOG_PATH="${RUNS_ROOT}/logs/${EXP_NAME}.log"

[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -d "${DATA_ROOT}/pusht_expert_train.lance" ]] || { echo "ERROR: PushT Lance dataset not found" >&2; exit 1; }
mkdir -p "${RUNS_ROOT}/wandb" "${RUNS_ROOT}/logs"
cd "${OGBENCH_ROOT}/impls"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_DIR="${RUNS_ROOT}/wandb" \
"${OGBENCH_ROOT}/.venv/bin/python" main.py \
  --env_name=visual-lewm-pusht-expert-train-v0 \
  --dataset_path="${DATA_ROOT}/pusht_expert_train.lance" \
  --agent=agents/hiql.py \
  --agent.batch_size=256 \
  --agent.lr=1e-4 \
  --agent.encoder=impala_small \
  --agent.subgoal_steps=10 \
  --agent.high_alpha=1.0 \
  --agent.low_alpha=3.0 \
  --agent.expectile=0.7 \
  --agent.tau=0.005 \
  --agent.rep_dim=10 \
  --agent.low_actor_rep_grad=True \
  --agent.p_aug=0.2 \
  --train_steps=100000 \
  --save_dir="${RUNS_ROOT}" \
  --log_interval=5000 \
  --save_interval=100000 \
  --run_group=lewm-pusht-visual-hiql-stable-bs256-s100k \
  --wandb_mode=online \
  --seed=0 \
  --eval_episodes=0 \
  --video_episodes=0 \
  2>&1 | tee "${LOG_PATH}"
