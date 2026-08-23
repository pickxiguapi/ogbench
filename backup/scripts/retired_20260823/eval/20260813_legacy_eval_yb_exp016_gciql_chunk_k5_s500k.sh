#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 TASK_KEY GPU_ID" >&2
  echo "TASK_KEY: antlarge | antgiant | humedium | cubesingle | cubedouble | scene | puzzle3x3" >&2
  exit 2
fi

TASK_KEY="$1"
GPU_ID="$2"
OGBENCH_ROOT="/root/data/yyf/ogbench"
RUNS_ROOT="/root/data/yyf/ogbench-native-runs"
EGL_LIB_DIR="/root/data/yyf/egl-runtime/root/usr/lib/x86_64-linux-gnu"
CHECKPOINT_STEP=500000
SEED=42
NUM_EVAL=50

case "${TASK_KEY}" in
  antlarge)
    ENV_NAME="visual-antmaze-large-navigate-v0"
    RUN_GROUP="EXP016_GCIQLChunk_antlarge_k5"
    ALPHA=0.3
    DISCOUNT=0.99
    P_AUG=0.0
    ;;
  antgiant)
    ENV_NAME="visual-antmaze-giant-navigate-v0"
    RUN_GROUP="EXP016_GCIQLChunk_antgiant_k5"
    ALPHA=0.3
    DISCOUNT=0.995
    P_AUG=0.0
    ;;
  humedium)
    ENV_NAME="visual-humanoidmaze-medium-navigate-v0"
    RUN_GROUP="EXP016_GCIQLChunk_humedium_k5"
    ALPHA=0.1
    DISCOUNT=0.995
    P_AUG=0.0
    ;;
  cubesingle)
    ENV_NAME="visual-cube-single-play-v0"
    RUN_GROUP="EXP016_GCIQLChunk_cubesingle_k5"
    ALPHA=1.0
    DISCOUNT=0.99
    P_AUG=0.5
    ;;
  cubedouble)
    ENV_NAME="visual-cube-double-play-v0"
    RUN_GROUP="EXP016_GCIQLChunk_cubedouble_k5"
    ALPHA=1.0
    DISCOUNT=0.99
    P_AUG=0.5
    ;;
  scene)
    ENV_NAME="visual-scene-play-v0"
    RUN_GROUP="EXP016_GCIQLChunk_scene_k5"
    ALPHA=1.0
    DISCOUNT=0.99
    P_AUG=0.5
    ;;
  puzzle3x3)
    ENV_NAME="visual-puzzle-3x3-play-v0"
    RUN_GROUP="EXP016_GCIQLChunk_puzzle3x3_k5"
    ALPHA=1.0
    DISCOUNT=0.99
    P_AUG=0.5
    ;;
  *)
    echo "ERROR: unknown TASK_KEY=${TASK_KEY}" >&2
    exit 2
    ;;
esac

CHECKPOINT_DIR="${RUNS_ROOT}/OGBench/${RUN_GROUP}/sd000_20260813_114337"
EVAL_ROOT="${RUNS_ROOT}/eval/EXP016/${TASK_KEY}/seed_${SEED}"

[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/params_${CHECKPOINT_STEP}.pkl" ]] || { echo "ERROR: checkpoint not found" >&2; exit 1; }
[[ -s "${CHECKPOINT_DIR}/flags.json" ]] || { echo "ERROR: flags.json not found" >&2; exit 1; }
[[ -s "${EGL_LIB_DIR}/libEGL.so.1" ]] || { echo "ERROR: user EGL runtime not found" >&2; exit 1; }

mkdir -p "${EVAL_ROOT}" "${RUNS_ROOT}/tmp" "${RUNS_ROOT}/wandb"
cd "${OGBENCH_ROOT}/impls"

MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
EGL_PLATFORM=surfaceless \
LD_LIBRARY_PATH="${EGL_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
TMPDIR="${RUNS_ROOT}/tmp" \
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_DIR="${RUNS_ROOT}/wandb" \
"${OGBENCH_ROOT}/.venv/bin/python" main.py \
  --env_name="${ENV_NAME}" \
  --agent=agents/gciql_chunk.py \
  --agent.actor_loss=ddpgbc \
  --agent.alpha="${ALPHA}" \
  --agent.chunk_size=5 \
  --agent.batch_size=256 \
  --agent.lr=3e-4 \
  --agent.discount="${DISCOUNT}" \
  --agent.expectile=0.9 \
  --agent.tau=0.005 \
  --agent.encoder=impala_small \
  --agent.p_aug="${P_AUG}" \
  --restore_path="${CHECKPOINT_DIR}" \
  --restore_epoch="${CHECKPOINT_STEP}" \
  --eval_only \
  --save_dir="${EVAL_ROOT}" \
  --run_group="EXP016_eval_${TASK_KEY}_s500k_seed${SEED}" \
  --wandb_mode=offline \
  --seed="${SEED}" \
  --eval_episodes="${NUM_EVAL}" \
  --eval_on_cpu=0 \
  --video_episodes=0 \
  2>&1 | tee "${EVAL_ROOT}/eval.log"

EVAL_CSV="$(find "${EVAL_ROOT}" -type f -name eval.csv -print | sort | tail -1)"
[[ -n "${EVAL_CSV}" && -s "${EVAL_CSV}" ]] || { echo "ERROR: eval.csv not produced" >&2; exit 1; }

if [[ -n "${EXPERIMENT_RECORDER_ROOT:-}" && -n "${EXPERIMENT_RUN_ID:-}" ]]; then
  python3 "${EXPERIMENT_RECORDER_ROOT}/scripts/aggregate_evals.py" \
    --run-id "${EXPERIMENT_RUN_ID}" \
    --database "${EXPERIMENT_RECORDER_ROOT}/data/experiments.json" \
    --events "${EXPERIMENT_RECORDER_ROOT}/data/run_events.csv" \
    --catalog "${EXPERIMENT_RECORDER_ROOT}/data/experiment_catalog.json" \
    "${EVAL_CSV}"
fi

echo "Evaluation completed: ${TASK_KEY}"
echo "Metrics: ${EVAL_CSV}"
