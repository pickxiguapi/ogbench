#!/usr/bin/env bash
set -euo pipefail

# Train the missing H25 goalmax25 History-MLP and Endpoint-Flow checkpoints,
# stage the canonical goalmax25 LatentPathFlow, then run the same full-episode
# ACID/success comparison used for H50 (50 episodes x eval seeds 0/1/42).
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

MODE=${MODE:-launch}
SESSION=${SESSION:-acid-subgoal-generators-h25-full-ep50}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
ROOT=/data-training/yyf/ogbench-lewm-policy-runs
H25_ROOT=${H25_ROOT:-$ROOT/latent-predictor-h25-goalmax25-ablation}
LPF_ROOT=${LPF_ROOT:-$ROOT/latent-path-flow-k10-goalmax25}
VIEW_ROOT=${VIEW_ROOT:-$LPF_ROOT/goalmax25_h25_predictor_ablation_view}
EVAL_ROOT=${EVAL_ROOT:-$ROOT/evals/lewm-4tasks/20260907_h25_goalmax25_subgoal_generator_success_only_lewmpp_policy777_ns1_cem300x5_h2_rh1_train0_eval0-1-42_ep50}
TMP_ROOT=${TMP_ROOT:-$ROOT/tmp/20260907-h25-goalmax25-subgoal-generator-success-only}
DRIVER_LOG=${DRIVER_LOG:-$EVAL_ROOT/driver.log}

tasks=(cube pusht reacher tworoom)
lewm_seeds=(3072 666 3072 3072)

stage_latent_path_flow() {
  local i task source target
  mkdir -p "$H25_ROOT"
  for i in "${!tasks[@]}"; do
    task=${tasks[$i]}
    source="$LPF_ROOT/latent_pathflow_${task}_lewm${lewm_seeds[$i]}_hist3_sg10_ab5_goalstride5_goalmax25_cfm_ns8_n200000_b1024_s0"
    target="$H25_ROOT/h25_goalmax25_latent_path_flow_${task}_lewm${lewm_seeds[$i]}_hist3_k10_pmatch18m_n200000_b1024_s0"
    test -s "$source/checkpoint_200000.msgpack"
    if [[ -L "$target" ]]; then
      [[ $(readlink -f "$target") == $(readlink -f "$source") ]] || {
        echo "LatentPathFlow link points elsewhere: $target" >&2; exit 3;
      }
    elif [[ -e "$target" ]]; then
      echo "LatentPathFlow target exists and is not a symlink: $target" >&2
      exit 3
    else
      ln -s "$source" "$target"
    fi
  done
}

driver() {
  mkdir -p "$H25_ROOT" "$EVAL_ROOT" "$TMP_ROOT"
  GPU_IDS="$GPU_IDS" ARCHITECTURES="history_mlp endpoint_flow" TRAIN_SEEDS=0 \
    HORIZON_TAG=h25_goalmax25 GOAL_OFFSET=25 GOAL_SAMPLING=aligned_future \
    MAX_GOAL_STEPS=25 RUNS_ROOT="$H25_ROOT" MANIFEST_ROOT="$H25_ROOT/manifests" \
    bash "$SCRIPT_DIR/20260904_train_node4_h50_predictor_ablation.sh"
  stage_latent_path_flow
  GPU_IDS="$GPU_IDS" WAIT_FOR_GPUS=0 \
    ARCHITECTURES="history_mlp endpoint_flow latent_path_flow" \
    TRAIN_SEEDS=0 EVAL_SEEDS="0 1 42" TRAIN_STEPS=200000 \
    NUM_EVAL=50 POLICY_GUIDANCE=mode GUIDANCE_GOAL_MODE=final \
    POLICY_SEED=777 POLICY_STEPS=100000 CEM_ITERATIONS=5 \
    HORIZON_TAG=h25_goalmax25 RUN_TAG=h25_goalmax25 \
    GENERATOR_FAMILY=goalmax25 GOAL_OFFSET_STEPS=25 EVAL_BUDGET=50 \
    RUNS_ROOT="$H25_ROOT" EVAL_ROOT="$EVAL_ROOT" TMP_ROOT="$TMP_ROOT" \
    bash "$OGBENCH_ROOT/exp/eval/lewm_4tasks/20260904_eval_node4_h50_predictor_ablation.sh"
  "$PYTHON_BIN" "$OGBENCH_ROOT/impls/aggregate_subgoal_success.py" \
    --root="$EVAL_ROOT" --prefix=h25_goalmax25 \
    --architectures history_mlp endpoint_flow latent_path_flow \
    --train-seed=0 --eval-seeds 0 1 42 \
    --tasks tworoom reacher pusht cube \
    --output="$EVAL_ROOT/aggregate_success.json"
  echo "DONE: $EVAL_ROOT/aggregate_success.json"
}

case "$MODE" in
  launch)
    mkdir -p "$EVAL_ROOT"
    tmux has-session -t "$SESSION" 2>/dev/null && {
      echo "tmux session already exists: $SESSION" >&2; exit 3;
    }
    printf -v command '%q ' env MODE=driver SESSION="$SESSION" GPU_IDS="$GPU_IDS" \
      H25_ROOT="$H25_ROOT" LPF_ROOT="$LPF_ROOT" VIEW_ROOT="$VIEW_ROOT" \
      EVAL_ROOT="$EVAL_ROOT" TMP_ROOT="$TMP_ROOT" \
      bash exp/train/latent_subgoal/20260907_run_node4_h25_subgoal_generator_feasibility_full_episode.sh
    printf -v quoted_log '%q' "$DRIVER_LOG"
    tmux new-session -d -s "$SESSION" -c "$OGBENCH_ROOT" \
      "$command >$quoted_log 2>&1"
    echo "launched tmux=$SESSION log=$DRIVER_LOG"
    ;;
  driver) driver ;;
  status)
    tmux ls 2>/dev/null | grep "$SESSION" || true
    tail -n 30 "$DRIVER_LOG" 2>/dev/null || true
    ;;
  *) echo "MODE must be launch, driver, or status" >&2; exit 2 ;;
esac
