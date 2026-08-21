#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/home/yyf/ogbench}"
DASHBOARD_ROOT="${DASHBOARD_ROOT:-/home/yyf/experiment-dashboard}"
DATA_ROOT="${DATA_ROOT:-/data/yyf/H-LeWM/datasets}"
RUNS_ROOT="${RUNS_ROOT:-/data/yyf/H-LeWM/ogbench-runs}"
VENV_DIR="${VENV_DIR:-/data/yyf/H-LeWM/envs/ogbench}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SOURCE_GIT_COMMIT="${SOURCE_GIT_COMMIT:-unknown}"
RETRY_TAG="${RETRY_TAG:-r2_separate_output}"

[[ -f "${DASHBOARD_ROOT}/scripts/recorded_run.sh" ]] || { echo "ERROR: dashboard recorder unavailable" >&2; exit 1; }
[[ -x "${VENV_DIR}/bin/python" ]] || { echo "ERROR: OGBench environment unavailable" >&2; exit 1; }

tasks=(tworoom reacher pusht)
datasets=(tworoom.lance reacher.lance pusht_expert_train.lance)
gpus=(0 1 2)

for dataset in "${datasets[@]}"; do
  [[ -f "${DATA_ROOT}/${dataset}/.conversion_complete" ]] || {
    echo "ERROR: complete dataset not found: ${DATA_ROOT}/${dataset}" >&2
    exit 1
  }
done

for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  gpu="${gpus[$i]}"
  task_output_dir="${RUNS_ROOT}/${task}"
  session="s11-gciql-chunk-awr-${task}-r2"
  run_id="EXP-010-s11-${task}-${RUN_STAMP}"
  exp_name="GCIQLChunkAWR_ogbench_lewm_${task}_k5_bs256_s100k_seed0_alpha3_expectile09_aug05_s11_${RETRY_TAG}"
  payload="{\"dataset\":\"${DATA_ROOT}/${datasets[$i]}\",\"output_dir\":\"${task_output_dir}\",\"source_repository\":\"https://github.com/pickxiguapi/ogbench\",\"source_git_commit\":\"${SOURCE_GIT_COMMIT}\",\"seed\":0,\"parameters\":{\"algorithm\":\"GCIQL-Chunk-Gaussian\",\"environment\":\"${task}\",\"chunk_size\":5,\"actor_loss\":\"awr\",\"alpha\":3.0,\"batch_size\":256,\"train_steps\":100000,\"encoder\":\"impala_small\",\"expectile\":0.9,\"p_aug\":0.5,\"wandb_mode\":\"offline\",\"retry\":2,\"output_isolation\":\"task_save_dir_and_run_group\"}}"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi

  tmux new-session -d -s "${session}" \
    env CUDA_VISIBLE_DEVICES="${gpu}" \
      RUNS_ROOT="${RUNS_ROOT}" \
      EXPERIMENT_EXTRA_PAYLOAD_JSON="${payload}" \
      bash "${DASHBOARD_ROOT}/scripts/recorded_run.sh" \
        EXP-010 "${exp_name}" "${run_id}" -- \
        bash "${OGBENCH_ROOT}/scripts/train/20260812_legacy_train_s11_lewm_gciql_chunk_awr_s100k.sh" "${task}"
  echo "Started ${run_id} on GPU ${gpu}; output root: ${task_output_dir}; tmux: ${session}"
done
