#!/usr/bin/env bash
set -euo pipefail
OGBENCH_ROOT="/root/data/yyf/ogbench"
RUNS_ROOT="/root/data/yyf/ogbench-native-runs"
EGL_LIB_DIR="/root/data/yyf/egl-runtime/root/usr/lib/x86_64-linux-gnu"
GPU_ID=0
EXP_NAME="GCIQLChunkDDPGBC_ogbench_visual_antmaze_large_navigate_k5_bs256_s500k_seed0_alpha03_gamma099"
LOG_PATH="${RUNS_ROOT}/logs/${EXP_NAME}.log"
[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -f "${OGBENCH_ROOT}/impls/agents/gciql_chunk.py" ]] || { echo "ERROR: GCIQL-Chunk agent not found" >&2; exit 1; }
[[ -f "${EGL_LIB_DIR}/libEGL.so.1" ]] || { echo "ERROR: user EGL runtime not found" >&2; exit 1; }
mkdir -p "${RUNS_ROOT}/wandb" "${RUNS_ROOT}/logs" "${RUNS_ROOT}/tmp"
cd "${OGBENCH_ROOT}/impls"
MUJOCO_GL=egl LD_LIBRARY_PATH="${EGL_LIB_DIR}" TMPDIR="${RUNS_ROOT}/tmp" CUDA_VISIBLE_DEVICES="${GPU_ID}" XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_DIR="${RUNS_ROOT}/wandb" \
"${OGBENCH_ROOT}/.venv/bin/python" main.py \
  --env_name=visual-antmaze-large-navigate-v0 --agent=agents/gciql_chunk.py \
  --agent.actor_loss=ddpgbc --agent.alpha=0.3 --agent.chunk_size=5 --agent.batch_size=256 \
  --agent.lr=3e-4 --agent.discount=0.99 --agent.expectile=0.9 --agent.tau=0.005 --agent.encoder=impala_small \
  --train_steps=500000 --save_dir="${RUNS_ROOT}" --log_interval=5000 --eval_interval=100000 --save_interval=100000 \
  --run_group=EXP016_GCIQLChunk_antlarge_k5 --wandb_mode=online --seed=0 --eval_episodes=50 --eval_on_cpu=0 --video_episodes=0 \
  2>&1 | tee "${LOG_PATH}"
