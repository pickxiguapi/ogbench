#!/usr/bin/env bash
set -euo pipefail

# Search Cube and PushT H50 closed-loop traces for continuous publication-figure
# candidates where both LeWM imagination and the general-generator K10 subgoal
# match the same real t+10 future frames.
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

BASE_SCRIPT="$SCRIPT_DIR/20260906_eval_node4_general_h50_closed_loop_visual_trace.sh"
TORCH_PYTHON_BIN=${TORCH_PYTHON_BIN:-/data-training/yyf/envs/latent-geometry/bin/python}
NUM_EVAL=${NUM_EVAL:-48}
EVAL_SEED=${EVAL_SEED:-91042}
GOAL_OFFSET=50
EVAL_BUDGET=100
POLICY_SEED=777
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror}
SUBGOAL_ROOT=${SUBGOAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10}
DECODER_ROOT=${DECODER_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/lewm-visual-decoder/20260905_mixed666_3072_official_cnn_image_decoder_aligned_epoch10_snapshot}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/lewm-visual-decoder-eval/20260910_general_uniform_future_h50_figure_candidate_search_ep${NUM_EVAL}_seed${EVAL_SEED}}
CURATED_ROOT=${CURATED_ROOT:-$OUTPUT_ROOT/curated_continuous6}
MODE=${MODE:-launch}

specs=(
  "cube:4:3072:/data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"
  "pusht:5:666:/data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack"
)

validate_checkpoint() {
  local task=$1 lewm_seed=$2
  local checkpoint="$SUBGOAL_ROOT/latent_pathflow_${task}_lewm${lewm_seed}_hist3_sg10_ab5_cfm_ns8_n200000_b1024_s0/checkpoint_200000.msgpack"
  "$PYTHON_BIN" - "$checkpoint" <<'PY'
import json
import pathlib
import sys

checkpoint = pathlib.Path(sys.argv[1])
config_path = checkpoint.parent / 'config.json'
if not checkpoint.is_file() or not config_path.is_file():
    raise SystemExit(f'missing generator checkpoint/config: {checkpoint}')
config = json.loads(config_path.read_text())
if config.get('goal_sampling') != 'hiql_uniform_future_same_trajectory':
    raise SystemExit(f'not a general_uniform_future generator: {checkpoint.parent}')
if config.get('max_goal_steps') is not None:
    raise SystemExit(f'bounded generator cannot be used at H50: {checkpoint.parent}')
print(f'verified general_uniform_future: {checkpoint.parent}')
PY
}

case "$MODE" in
  launch)
    mkdir -p "$OUTPUT_ROOT/logs"
    for spec in "${specs[@]}"; do
      IFS=: read -r task gpu lewm_seed lewm_checkpoint <<<"$spec"
      validate_checkpoint "$task" "$lewm_seed"
      if [[ -s "$OUTPUT_ROOT/$task/result.json" ]]; then
        echo "SKIP completed task=$task output=$OUTPUT_ROOT/$task/result.json"
        continue
      fi
      session="lewm-h50-figsearch-$task"
      tmux has-session -t "$session" 2>/dev/null && { echo "Session exists: $session" >&2; exit 3; }
      printf -v worker_cmd '%q ' env MODE=worker TASK="$task" GPU_ID="$gpu" \
        LEWM_SEED="$lewm_seed" LEWM_CHECKPOINT="$lewm_checkpoint" \
        PYTHON_BIN="$PYTHON_BIN" TORCH_PYTHON_BIN="$TORCH_PYTHON_BIN" \
        LEWM_DATA_ROOT="$LEWM_DATA_ROOT" POLICY_ROOT="$POLICY_ROOT" \
        SUBGOAL_ROOT="$SUBGOAL_ROOT" DECODER_ROOT="$DECODER_ROOT" \
        OUTPUT_ROOT="$OUTPUT_ROOT" NUM_EVAL="$NUM_EVAL" EVAL_SEED="$EVAL_SEED" \
        GOAL_OFFSET="$GOAL_OFFSET" EVAL_BUDGET="$EVAL_BUDGET" POLICY_SEED="$POLICY_SEED" \
        bash "$BASE_SCRIPT"
      tmux new-session -d -s "$session" -c "$OGBENCH_ROOT" "$worker_cmd"
      echo "launched task=$task gpu=$gpu session=$session"
    done
    ;;
  status)
    tmux list-sessions 2>/dev/null | grep 'lewm-h50-figsearch-' || true
    for task in cube pusht; do
      printf '%s ' "$task"
      [[ -s "$OUTPUT_ROOT/$task/result.json" ]] && echo COMPLETE || echo RUNNING_OR_PENDING
      [[ -f "$OUTPUT_ROOT/logs/$task.log" ]] && tail -n 3 "$OUTPUT_ROOT/logs/$task.log"
    done
    ;;
  rank)
    test -s "$OUTPUT_ROOT/cube/figures/manifest.json"
    test -s "$OUTPUT_ROOT/pusht/figures/manifest.json"
    "$TORCH_PYTHON_BIN" "$OGBENCH_ROOT/impls/rank_lewm_figure_candidates.py" \
      --input-root="$OUTPUT_ROOT" --output-root="$CURATED_ROOT" \
      --tasks cube pusht --window=6 --topk=12
    ;;
  *)
    echo "MODE must be launch, status, or rank" >&2
    exit 2
    ;;
esac
