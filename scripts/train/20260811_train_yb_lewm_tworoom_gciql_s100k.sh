#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="/root/data/yyf/ogbench"
DATA_ROOT="/root/data/yyf/stable-worldmodel/datasets"
RUNS_ROOT="/root/data/yyf/lewm-runs"
GPU_ID=4

[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -e "${DATA_ROOT}/tworoom.lance" ]] || { echo "ERROR: TwoRoom Lance dataset not found" >&2; exit 1; }
mkdir -p "${RUNS_ROOT}/wandb" "${RUNS_ROOT}/logs"
cd "${OGBENCH_ROOT}/impls"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_DIR="${RUNS_ROOT}/wandb" \
"${OGBENCH_ROOT}/.venv/bin/python" main.py \
  --env_name=visual-lewm-tworoom-v0 \
  --dataset_path="${DATA_ROOT}/tworoom.lance" \
  --agent=agents/gciql.py \
  --agent.alpha=1.0 \
  --agent.batch_size=256 \
  --agent.encoder=impala_small \
  --agent.p_aug=0.5 \
  --train_steps=100000 \
  --save_dir="${RUNS_ROOT}" \
  --log_interval=5000 \
  --save_interval=100000 \
  --run_group=lewm-tworoom-visual-gciql-bs256-100k \
  --wandb_mode=offline \
  --seed=0 \
  --eval_episodes=0 \
  --video_episodes=0 \
  2>&1 | tee "${RUNS_ROOT}/logs/gciql-ogbench-lewm-tworoom-bs256-s100k-seed0.log"
