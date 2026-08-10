#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMPLS_DIR="${IMPLS_DIR:-${REPO_ROOT}/impls}"
DATASETS_DIR="${DATASETS_DIR:-/root/data/yyf/stable-worldmodel/datasets}"
RUNS_ROOT="${RUNS_ROOT:-/root/data/yyf/lewm-runs}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
WANDB_MODE="${WANDB_MODE:-offline}"
GPU_IDS_CSV="${GPU_IDS:-5,6,7}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS_CSV}"
if [[ "${#GPU_ARRAY[@]}" -ne 3 ]]; then
  echo "ERROR: GPU_IDS must contain exactly three comma-separated GPU IDs." >&2
  exit 1
fi

[[ -x "${PYTHON_BIN}" ]] || {
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
}
[[ -d "${IMPLS_DIR}" ]] || {
  echo "ERROR: OGBench impls directory not found: ${IMPLS_DIR}" >&2
  exit 1
}

for dataset in tworoom.lance reacher.lance; do
  [[ -e "${DATASETS_DIR}/${dataset}" ]] || {
    echo "ERROR: dataset not found: ${DATASETS_DIR}/${dataset}" >&2
    exit 1
  }
done

mkdir -p "${RUNS_ROOT}/wandb" "${RUNS_ROOT}/logs"

names=(tworoom-gciql tworoom-hiql reacher-gciql)
env_names=(visual-lewm-tworoom-v0 visual-lewm-tworoom-v0 visual-lewm-reacher-v0)
datasets=(tworoom.lance tworoom.lance reacher.lance)
agents=(gciql hiql gciql)
groups=(
  lewm-tworoom-visual-gciql-bs256-100k
  lewm-tworoom-visual-hiql-bs256-100k
  lewm-reacher-visual-gciql-bs256-100k
)

for name in "${names[@]}"; do
  session="yb-ogbench-${name}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
done

for i in "${!names[@]}"; do
  name="${names[$i]}"
  session="yb-ogbench-${name}"
  log_file="${RUNS_ROOT}/logs/${session}.log"

  common_args=(
    "--env_name=${env_names[$i]}"
    "--dataset_path=${DATASETS_DIR}/${datasets[$i]}"
    "--agent=agents/${agents[$i]}.py"
    --agent.batch_size=256
    --agent.encoder=impala_small
    --agent.p_aug=0.5
    --train_steps=100000
    "--save_dir=${RUNS_ROOT}"
    --log_interval=5000
    --save_interval=100000
    "--run_group=${groups[$i]}"
    "--wandb_mode=${WANDB_MODE}"
    --eval_episodes=0
    --video_episodes=0
  )

  if [[ "${agents[$i]}" == "hiql" ]]; then
    common_args+=(
      --agent.high_alpha=3.0
      --agent.low_actor_rep_grad=True
      --agent.low_alpha=3.0
      --agent.subgoal_steps=10
    )
  else
    common_args+=(--agent.alpha=1.0)
  fi

  printf -v command '%q ' env \
    "CUDA_VISIBLE_DEVICES=${GPU_ARRAY[$i]}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "WANDB_DIR=${RUNS_ROOT}/wandb" \
    "${PYTHON_BIN}" main.py "${common_args[@]}"
  command+="2>&1 | tee $(printf '%q' "${log_file}")"

  tmux new-session -d -s "${session}" -c "${IMPLS_DIR}" "${command}"
  echo "Started ${session} on GPU ${GPU_ARRAY[$i]} (log: ${log_file})"
done

echo
tmux list-sessions | grep '^yb-ogbench-' || true
