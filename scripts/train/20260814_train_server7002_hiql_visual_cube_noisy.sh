#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/train/20260814_train_server7002_hiql_visual_cube_noisy.sh {single|double|triple}" >&2
  exit 2
fi

TASK="$1"
case "$TASK" in
  single|double|triple) ;;
  *) echo "ERROR: unsupported cube task: ${TASK}" >&2; exit 2 ;;
esac

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/yyf/ogbench}"
RUNS_ROOT="${RUNS_ROOT:-/mnt/18T/yyf/ogbench-native-runs/exp020_hiql_visual_cube_noisy}"
TMP_ROOT="${TMP_ROOT:-/mnt/18T/yyf/ogbench-tmp/exp020}"
ENV_NAME="visual-cube-${TASK}-noisy-v0"
RUN_GROUP="EXP020_HIQL_visual_cube_${TASK}_noisy_official_s500k"
LOG_PATH="${RUNS_ROOT}/logs/HIQL_ogbench_visual_cube_${TASK}_noisy_bs256_s500k_seed0_official.log"

[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || { echo "ERROR: OGBench Python not found" >&2; exit 1; }
[[ -f "${OGBENCH_ROOT}/impls/agents/hiql.py" ]] || { echo "ERROR: HIQL agent not found" >&2; exit 1; }
[[ ! -e "${RUNS_ROOT}/OGBench/${RUN_GROUP}" ]] || { echo "ERROR: run group already exists" >&2; exit 1; }
mkdir -p "${RUNS_ROOT}/logs" "${TMP_ROOT}"
cd "${OGBENCH_ROOT}/impls"

MUJOCO_GL=egl TMPDIR="${TMP_ROOT}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "${OGBENCH_ROOT}/.venv/bin/python" main.py \
  --env_name="${ENV_NAME}" --train_steps=500000 \
  --eval_episodes=50 --eval_on_cpu=0 --agent=agents/hiql.py \
  --agent.batch_size=256 --agent.encoder=impala_small \
  --agent.high_alpha=3.0 --agent.low_actor_rep_grad=True --agent.low_alpha=3.0 \
  --agent.p_aug=0.5 --agent.subgoal_steps=10 \
  --save_dir="${RUNS_ROOT}" --log_interval=5000 --eval_interval=100000 --save_interval=100000 \
  --run_group="${RUN_GROUP}" --wandb_mode=disabled --seed=0 --video_episodes=0 \
  2>&1 | tee "${LOG_PATH}"
