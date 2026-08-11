#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${OGBENCH_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DASHBOARD_ROOT="${DASHBOARD_ROOT:-/root/data/yyf/experiment-dashboard}"
DATASETS_DIR="${DATASETS_DIR:-/root/data/yyf/stable-worldmodel/datasets}"
RUNS_ROOT="${RUNS_ROOT:-/root/data/yyf/lewm-runs}"
GPU_IDS_CSV="${GPU_IDS:-3,7}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_STEPS="${TRAIN_STEPS:-100000}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"
SEED="${SEED:-0}"
EXPERIMENT_ID="${EXPERIMENT_ID:-EXP-011}"
LAUNCH_TAG="${LAUNCH_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

if (( TRAIN_STEPS % 1000 == 0 )); then
  TRAIN_LABEL="$((TRAIN_STEPS / 1000))k"
else
  TRAIN_LABEL="${TRAIN_STEPS}"
fi

TASKS=(pusht cube)
DATASETS=(pusht_expert_train.lance cube_single_expert.lance)
ENV_NAMES=(visual-lewm-pusht-expert-train-v0 visual-lewm-cube-single-expert-v0)
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS_CSV}"

[[ "${#GPU_ARRAY[@]}" -eq "${#TASKS[@]}" ]] || {
  echo "ERROR: GPU_IDS must contain exactly two comma-separated GPU IDs." >&2
  exit 1
}
[[ -x "${REPO_ROOT}/.venv/bin/python" ]] || {
  echo "ERROR: OGBench Python not found: ${REPO_ROOT}/.venv/bin/python" >&2
  exit 1
}
[[ -f "${DASHBOARD_ROOT}/scripts/recorded_run.sh" ]] || {
  echo "ERROR: experiment recorder not found: ${DASHBOARD_ROOT}/scripts/recorded_run.sh" >&2
  exit 1
}

for dataset in "${DATASETS[@]}"; do
  [[ -e "${DATASETS_DIR}/${dataset}" ]] || {
    echo "ERROR: dataset not found: ${DATASETS_DIR}/${dataset}" >&2
    exit 1
  }
done

for task in "${TASKS[@]}"; do
  session="yb-gciql-chunk-exp002-${task}-${LAUNCH_TAG}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
done

mkdir -p "${RUNS_ROOT}/logs"

for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  gpu="${GPU_ARRAY[$i]}"
  session="yb-gciql-chunk-exp002-${task}-${LAUNCH_TAG}"
  run_id="${EXPERIMENT_ID}-${task}-${LAUNCH_TAG}"
  exp_name="GCIQLChunk_ogbench_${task}_gaussian_ddpgbc_k${CHUNK_SIZE}_bs${BATCH_SIZE}_s${TRAIN_LABEL}_seed${SEED}"
  log_file="${RUNS_ROOT}/logs/${session}.log"
  payload="$(ENV_NAME="${ENV_NAMES[$i]}" DATASET="${DATASETS_DIR}/${DATASETS[$i]}" LOG_FILE="${log_file}" RUNS_ROOT="${RUNS_ROOT}" BATCH_SIZE="${BATCH_SIZE}" TRAIN_STEPS="${TRAIN_STEPS}" CHUNK_SIZE="${CHUNK_SIZE}" SEED="${SEED}" python3 -c 'import json,os; print(json.dumps({"dataset":os.environ["DATASET"],"log_path":os.environ["LOG_FILE"],"output_dir":os.environ["RUNS_ROOT"],"seed":int(os.environ["SEED"]),"parameters":{"algorithm":"GCIQL-Chunk-Gaussian","baseline":"EXP-002 OGBench GCIQL","environment":os.environ["ENV_NAME"],"chunk_size":int(os.environ["CHUNK_SIZE"]),"actor_loss":"ddpgbc","alpha":1.0,"batch_size":int(os.environ["BATCH_SIZE"]),"train_steps":int(os.environ["TRAIN_STEPS"]),"encoder":"impala_small","p_aug":0.5}},ensure_ascii=False))')"

  printf -v command '%q ' env \
    "CUDA_VISIBLE_DEVICES=${gpu}" \
    "BATCH_SIZE=${BATCH_SIZE}" \
    "TRAIN_STEPS=${TRAIN_STEPS}" \
    "CHUNK_SIZE=${CHUNK_SIZE}" \
    "SEED=${SEED}" \
    "DATASETS_DIR=${DATASETS_DIR}" \
    "RUNS_ROOT=${RUNS_ROOT}" \
    "EXPERIMENT_EXTRA_PAYLOAD_JSON=${payload}" \
    bash "${DASHBOARD_ROOT}/scripts/recorded_run.sh" \
      "${EXPERIMENT_ID}" "${exp_name}" "${run_id}" -- \
      bash "${SCRIPT_DIR}/0811_yb_train_lewm_gciql_chunk_task.sh" "${task}"
  command+="2>&1 | tee $(printf '%q' "${log_file}")"

  tmux new-session -d -s "${session}" -c "${REPO_ROOT}" "${command}"
  echo "Started ${task}: session=${session} GPU=${gpu} run_id=${run_id} log=${log_file}"
done

echo
echo "EXP-011 launched against EXP-002 with DDPG+BC alpha=1."
tmux list-sessions | grep "^yb-gciql-chunk-exp002-.*-${LAUNCH_TAG}:" || true
