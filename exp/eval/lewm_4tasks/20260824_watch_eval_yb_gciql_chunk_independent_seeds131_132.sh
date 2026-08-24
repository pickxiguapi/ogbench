#!/usr/bin/env bash
set -euo pipefail

# 英博云：等待 independent GCIQL-Chunk seed131/132 全部训练完成，再自动评测 CEM-only、policy-only 和 CEM+policy。
CLIENT_ID=${CLIENT_ID:-yb}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
POLICY_STEPS=${POLICY_STEPS:-100000}
POLICY_BATCH_SIZE=${POLICY_BATCH_SIZE:-256}
P_AUG=${P_AUG:-0.5}
LEWM_EPOCH=${LEWM_EPOCH:-10}
LEWM_SEED=${LEWM_SEED:-3072}
EVAL_SEED=${EVAL_SEED:-42}
NUM_EVAL=${NUM_EVAL:-50}
POLL_SECONDS=${POLL_SECONDS:-300}
EVAL_ROOT=${EVAL_ROOT:-/root/data/yyf/lewm-final/evals/lewm-4tasks/20260824_gciql_chunk_independent_seeds131_132}
EXECUTOR="$SCRIPT_DIR/20260823_eval_yb_lewm_4tasks.sh"
source "$OGBENCH_ROOT/scripts/client_env.sh"

seeds=(131 132)
tags=(cube pusht reacher tworoom)

checkpoint_path() {
  local seed=$1
  local tag=$2
  echo "$GCIQL_RUNS_ROOT/gciql-chunk-4tasks/gc4_${tag}_ind_n${POLICY_STEPS}_b${POLICY_BATCH_SIZE}_a${P_AUG}_sd${seed}/params_${POLICY_STEPS}.pkl"
}

echo "[$(date '+%F %T %Z')] waiting for eight final policy checkpoints"
while true; do
  missing=()
  for seed in "${seeds[@]}"; do
    for tag in "${tags[@]}"; do
      checkpoint=$(checkpoint_path "$seed" "$tag")
      [[ -s "$checkpoint" ]] || missing+=("$checkpoint")
    done
  done

  if (( ${#missing[@]} == 0 )); then
    if ! pgrep -f '[t]rain_gciql_chunk.py.*--seed=(131|132)' >/dev/null; then
      break
    fi
  elif ! pgrep -f '[t]rain_gciql_chunk.py.*--seed=(131|132)' >/dev/null; then
    echo "[$(date '+%F %T %Z')] ERROR: training stopped before all final checkpoints were written" >&2
    printf 'missing: %s\n' "${missing[@]}" >&2
    exit 1
  fi

  echo "[$(date '+%F %T %Z')] ${#missing[@]} checkpoints still missing"
  sleep "$POLL_SECONDS"
done

echo "[$(date '+%F %T %Z')] checkpoints complete; starting CEM-only evaluation"
CLIENT_ID="$CLIENT_ID" MODE=lewm REPRESENTATION_MODE=independent \
  POLICY_STEPS="$POLICY_STEPS" POLICY_BATCH_SIZE="$POLICY_BATCH_SIZE" P_AUG="$P_AUG" \
  LEWM_EPOCH="$LEWM_EPOCH" LEWM_SEED="$LEWM_SEED" EVAL_SEED="$EVAL_SEED" NUM_EVAL="$NUM_EVAL" \
  GPU_IDS="0 1 2 3" OUTPUT_ROOT="$EVAL_ROOT/cem_only" \
  bash "$EXECUTOR"

echo "[$(date '+%F %T %Z')] starting policy-only evaluations for seeds 131 and 132"
CLIENT_ID="$CLIENT_ID" MODE=policy REPRESENTATION_MODE=independent POLICY_SEED=131 \
  POLICY_STEPS="$POLICY_STEPS" POLICY_BATCH_SIZE="$POLICY_BATCH_SIZE" P_AUG="$P_AUG" \
  LEWM_EPOCH="$LEWM_EPOCH" LEWM_SEED="$LEWM_SEED" EVAL_SEED="$EVAL_SEED" NUM_EVAL="$NUM_EVAL" \
  GPU_IDS="0 1 2 3" OUTPUT_ROOT="$EVAL_ROOT/policy_seed131" \
  bash "$EXECUTOR" &
policy_131_pid=$!
CLIENT_ID="$CLIENT_ID" MODE=policy REPRESENTATION_MODE=independent POLICY_SEED=132 \
  POLICY_STEPS="$POLICY_STEPS" POLICY_BATCH_SIZE="$POLICY_BATCH_SIZE" P_AUG="$P_AUG" \
  LEWM_EPOCH="$LEWM_EPOCH" LEWM_SEED="$LEWM_SEED" EVAL_SEED="$EVAL_SEED" NUM_EVAL="$NUM_EVAL" \
  GPU_IDS="4 5 6 7" OUTPUT_ROOT="$EVAL_ROOT/policy_seed132" \
  bash "$EXECUTOR" &
policy_132_pid=$!
policy_status=0
wait "$policy_131_pid" || policy_status=$?
wait "$policy_132_pid" || policy_status=$?
if (( policy_status != 0 )); then
  echo "[$(date '+%F %T %Z')] ERROR: at least one policy-only evaluation failed" >&2
  exit "$policy_status"
fi

echo "[$(date '+%F %T %Z')] starting CEM+policy evaluations for seeds 131 and 132"
CLIENT_ID="$CLIENT_ID" MODE=guided REPRESENTATION_MODE=independent POLICY_SEED=131 \
  POLICY_STEPS="$POLICY_STEPS" POLICY_BATCH_SIZE="$POLICY_BATCH_SIZE" P_AUG="$P_AUG" \
  LEWM_EPOCH="$LEWM_EPOCH" LEWM_SEED="$LEWM_SEED" EVAL_SEED="$EVAL_SEED" NUM_EVAL="$NUM_EVAL" \
  GPU_IDS="0 1 2 3" OUTPUT_ROOT="$EVAL_ROOT/guided_seed131" \
  bash "$EXECUTOR" &
guided_131_pid=$!
CLIENT_ID="$CLIENT_ID" MODE=guided REPRESENTATION_MODE=independent POLICY_SEED=132 \
  POLICY_STEPS="$POLICY_STEPS" POLICY_BATCH_SIZE="$POLICY_BATCH_SIZE" P_AUG="$P_AUG" \
  LEWM_EPOCH="$LEWM_EPOCH" LEWM_SEED="$LEWM_SEED" EVAL_SEED="$EVAL_SEED" NUM_EVAL="$NUM_EVAL" \
  GPU_IDS="4 5 6 7" OUTPUT_ROOT="$EVAL_ROOT/guided_seed132" \
  bash "$EXECUTOR" &
guided_132_pid=$!
guided_status=0
wait "$guided_131_pid" || guided_status=$?
wait "$guided_132_pid" || guided_status=$?
if (( guided_status != 0 )); then
  echo "[$(date '+%F %T %Z')] ERROR: at least one CEM+policy evaluation failed" >&2
  exit "$guided_status"
fi

echo "[$(date '+%F %T %Z')] all evaluations complete: $EVAL_ROOT"
