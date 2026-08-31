#!/usr/bin/env bash
set -euo pipefail

# A800 node4：默认 GPU0–3 并行训练 TwoRoom/PushT/Cube/Reacher 四个 3 帧历史 K5/K10 LatentPathFlow；frozen z192、仅 CFM loss、200k、bs1024、seed0。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
TRAIN_STEPS=${TRAIN_STEPS:-200000}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
TRAIN_SEED=${TRAIN_SEED:-0}
SUBGOAL_STEPS=${SUBGOAL_STEPS:-10}
ACTION_BLOCK=${ACTION_BLOCK:-5}
LEWM_LATENT_ROOT=${LEWM_LATENT_ROOT:-/data-training/yyf/datasets/lewm-latents}
SUBGOAL_RUNS_ROOT=${SUBGOAL_RUNS_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10}
TASKS=${TASKS:-"tworoom pusht cube reacher"}

run_task() {
  local task=$1
  local gpu_id latent_dataset exp_name
  case "$task" in
    tworoom)
      gpu_id=0
      latent_dataset="$LEWM_LATENT_ROOT/tworoom__lewm_s3072_e10_z192.h5"
      exp_name="latent_pathflow_tworoom_lewm3072_hist3_sg${SUBGOAL_STEPS}_ab${ACTION_BLOCK}_cfm_n${TRAIN_STEPS}_b${TRAIN_BATCH_SIZE}_s${TRAIN_SEED}"
      ;;
    pusht)
      gpu_id=1
      latent_dataset="$LEWM_LATENT_ROOT/pusht_expert_train__lewm_s666_e10_z192.h5"
      exp_name="latent_pathflow_pusht_lewm666_hist3_sg${SUBGOAL_STEPS}_ab${ACTION_BLOCK}_cfm_n${TRAIN_STEPS}_b${TRAIN_BATCH_SIZE}_s${TRAIN_SEED}"
      ;;
    cube)
      gpu_id=2
      latent_dataset="$LEWM_LATENT_ROOT/cube_single_expert__lewm_s3072_e10_z192.h5"
      exp_name="latent_pathflow_cube_lewm3072_hist3_sg${SUBGOAL_STEPS}_ab${ACTION_BLOCK}_cfm_n${TRAIN_STEPS}_b${TRAIN_BATCH_SIZE}_s${TRAIN_SEED}"
      ;;
    reacher)
      gpu_id=3
      latent_dataset="$LEWM_LATENT_ROOT/reacher__lewm_s3072_e10_z192.h5"
      exp_name="latent_pathflow_reacher_lewm3072_hist3_sg${SUBGOAL_STEPS}_ab${ACTION_BLOCK}_cfm_n${TRAIN_STEPS}_b${TRAIN_BATCH_SIZE}_s${TRAIN_SEED}"
      ;;
    *)
      echo "Unsupported TASKS entry: $task" >&2
      return 1
      ;;
  esac

  CLIENT_ID="$CLIENT_ID" \
  GPU_ID="$gpu_id" \
  TRAIN_STEPS="$TRAIN_STEPS" \
  TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
  TRAIN_SEED="$TRAIN_SEED" \
  SUBGOAL_STEPS="$SUBGOAL_STEPS" \
  ACTION_BLOCK="$ACTION_BLOCK" \
  LATENT_DATASET="$latent_dataset" \
  SUBGOAL_RUNS_ROOT="$SUBGOAL_RUNS_ROOT" \
  EXP_NAME="$exp_name" \
  bash "$SCRIPT_DIR/20260831_train_latent_path_flow_k10_common.sh" &
}

for task in $TASKS; do
  run_task "$task"
done
wait
