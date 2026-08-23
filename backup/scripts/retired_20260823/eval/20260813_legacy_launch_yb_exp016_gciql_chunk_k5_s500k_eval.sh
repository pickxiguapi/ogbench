#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 GPU_ID TASK_KEY [TASK_KEY ...]" >&2
  exit 2
fi

GPU_ID="$1"
shift

OGBENCH_ROOT="/root/data/yyf/ogbench"
DASHBOARD_ROOT="${EXPERIMENT_DASHBOARD_ROOT:-/root/data/yyf/experiment-dashboard}"
EVAL_SCRIPT="${OGBENCH_ROOT}/scripts/eval/20260813_legacy_eval_yb_exp016_gciql_chunk_k5_s500k.sh"
RECORDED_RUN="${DASHBOARD_ROOT}/scripts/recorded_run.sh"

[[ -f "${EVAL_SCRIPT}" ]] || { echo "ERROR: eval Bash not found: ${EVAL_SCRIPT}" >&2; exit 1; }
[[ -f "${RECORDED_RUN}" ]] || { echo "ERROR: recorder not found: ${RECORDED_RUN}" >&2; exit 1; }

for TASK_KEY in "$@"; do
  case "${TASK_KEY}" in
    antlarge)
      RUN_ID="EXP-016-cs-3ab64-052a9-server-20260813T034328Z-56989"
      EXP_NAME="GCIQLChunkDDPGBC_ogbench_visual_antmaze_large_navigate_k5_bs256_s500k_seed0_alpha03_gamma099"
      ;;
    antgiant)
      RUN_ID="EXP-016-cs-3ab64-052a9-server-20260813T034328Z-56992"
      EXP_NAME="GCIQLChunkDDPGBC_ogbench_visual_antmaze_giant_navigate_k5_bs256_s500k_seed0_alpha03_gamma0995"
      ;;
    humedium)
      RUN_ID="EXP-016-cs-3ab64-052a9-server-20260813T034328Z-56999"
      EXP_NAME="GCIQLChunkDDPGBC_ogbench_visual_humanoidmaze_medium_navigate_k5_bs256_s500k_seed0_alpha01_gamma0995"
      ;;
    cubesingle)
      RUN_ID="EXP-016-cs-3ab64-052a9-server-20260813T034328Z-57010"
      EXP_NAME="GCIQLChunkDDPGBC_ogbench_visual_cube_single_play_k5_bs256_s500k_seed0_alpha1_gamma099_aug05"
      ;;
    cubedouble)
      RUN_ID="EXP-016-cs-3ab64-052a9-server-20260813T034328Z-57021"
      EXP_NAME="GCIQLChunkDDPGBC_ogbench_visual_cube_double_play_k5_bs256_s500k_seed0_alpha1_gamma099_aug05"
      ;;
    scene)
      RUN_ID="EXP-016-cs-3ab64-052a9-server-20260813T034328Z-57031"
      EXP_NAME="GCIQLChunkDDPGBC_ogbench_visual_scene_play_k5_bs256_s500k_seed0_alpha1_gamma099_aug05"
      ;;
    puzzle3x3)
      RUN_ID="EXP-016-cs-3ab64-052a9-server-20260813T034328Z-57041"
      EXP_NAME="GCIQLChunkDDPGBC_ogbench_visual_puzzle_3x3_play_k5_bs256_s500k_seed0_alpha1_gamma099_aug05"
      ;;
    *)
      echo "ERROR: unknown TASK_KEY=${TASK_KEY}" >&2
      exit 2
      ;;
  esac

  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  export EXPERIMENT_EXTRA_PAYLOAD_JSON="{\"evaluation_seed\":42,\"checkpoint_step\":500000,\"eval_episodes_per_task\":50,\"environment_key\":\"${TASK_KEY}\"}"
  bash "${RECORDED_RUN}" EXP-016 "${EXP_NAME}" "${RUN_ID}" \
    --eval-only bash "${EVAL_SCRIPT}" "${TASK_KEY}" "${GPU_ID}"
done
