#!/usr/bin/env bash
set -euo pipefail

# Reproduce GC-IDM on the four frozen-LeWM tasks under the LeWM++ long-range
# protocol: goal offset H in {25, 50, 75, 100}, evaluation budget 2H.
# Defaults: 50 episodes per cell, evaluation seed 42, official GC-IDM commit.

REMOTE_HOST=${REMOTE_HOST:-a800-node4}
SESSION_NAME=${SESSION_NAME:-gcidm_h25_50_75_100}
NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
GCIDM_COMMIT=${GCIDM_COMMIT:-48c45b1cb2b34dd2c1c61d222c8309de567fde55}
REMOTE_SCRIPT=${REMOTE_SCRIPT:-/data-training/yyf/experiments/gcidm/20260903_run_node4_gcidm_h25_50_75_100.sh}

for value in "$NUM_EVAL" "$EVAL_SEED"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "NUM_EVAL and EVAL_SEED must be non-negative integers" >&2; exit 2; }
done
[[ "$GCIDM_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "GCIDM_COMMIT must be a full 40-character SHA" >&2; exit 2; }

if [[ ${1:-} != "--remote" ]]; then
  script_dir=$(dirname "$REMOTE_SCRIPT")
  ssh "$REMOTE_HOST" "mkdir -p '$script_dir'"
  rsync -a "$0" "$REMOTE_HOST:$REMOTE_SCRIPT"
  ssh "$REMOTE_HOST" "
    set -euo pipefail
    if screen -list | grep -q '[.]${SESSION_NAME}[[:space:]]'; then
      echo 'screen session ${SESSION_NAME} is already running' >&2
      exit 3
    fi
    screen -dmS '${SESSION_NAME}' env \
      NUM_EVAL='${NUM_EVAL}' EVAL_SEED='${EVAL_SEED}' GCIDM_COMMIT='${GCIDM_COMMIT}' \
      bash '$REMOTE_SCRIPT' --remote
    echo 'launched screen session ${SESSION_NAME}'
  "
  exit 0
fi

SOURCE_REPO=/data-training/yyf/src/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models
EVAL_REPO="/data-training/yyf/src/gcidm-eval-${GCIDM_COMMIT:0:8}"
PYTHON=/data-training/yyf/envs/latent-geometry/bin/python
STABLEWM_HOME=/data-training/yyf/latent-geometry
OUTPUT_ROOT="/data-training/yyf/outputs/latent-geometry/eval-long-horizon/gcidm_official_${GCIDM_COMMIT:0:8}_ep${NUM_EVAL}_seed${EVAL_SEED}_budget2h"

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$OUTPUT_ROOT/launcher.log") 2>&1

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "commit=$GCIDM_COMMIT"
echo "num_eval=$NUM_EVAL"
echo "eval_seed=$EVAL_SEED"
echo "protocol=goal_offset_H,eval_budget_2H"

git -C "$SOURCE_REPO" fetch origin master
remote_commit=$(git -C "$SOURCE_REPO" rev-parse origin/master)
if [[ "$remote_commit" != "$GCIDM_COMMIT" ]]; then
  echo "official origin/master moved: expected $GCIDM_COMMIT, found $remote_commit" >&2
  exit 4
fi

if [[ ! -d "$EVAL_REPO/.git" && ! -f "$EVAL_REPO/.git" ]]; then
  git -C "$SOURCE_REPO" worktree add --detach "$EVAL_REPO" "$GCIDM_COMMIT"
fi
[[ $(git -C "$EVAL_REPO" rev-parse HEAD) == "$GCIDM_COMMIT" ]]
[[ -z $(git -C "$EVAL_REPO" status --short) ]] || { echo "clean evaluation worktree is dirty" >&2; exit 5; }

declare -A idm_paths=(
  [tworoom]=/data-training/yyf/outputs/latent-geometry/tworoom/tworoom_gcidm.pt
  [reacher]=/data-training/yyf/outputs/latent-geometry/reacher/reacher_gcidm.pt
  [pusht]=/data-training/yyf/outputs/latent-geometry/pusht/pusht_gcidm.pt
  [cube]=/data-training/yyf/outputs/latent-geometry/cube/cube_gcidm.pt
)

for task in tworoom reacher pusht cube; do
  [[ -s ${idm_paths[$task]} ]] || { echo "missing checkpoint: ${idm_paths[$task]}" >&2; exit 6; }
done

export STABLEWM_HOME
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

run_cell() {
  local task=$1
  local horizon=$2
  local gpu=$3
  local budget=$((2 * horizon))
  local log="$OUTPUT_ROOT/${task}_h${horizon}_b${budget}.log"

  echo "launch task=$task H=$horizon budget=$budget gpu=$gpu log=$log"
  (
    cd "$EVAL_REPO"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u eval_idm.py \
      --dataset "$task" \
      --idm "${idm_paths[$task]}" \
      --num-eval "$NUM_EVAL" \
      --goal-offset "$horizon" \
      --eval-budget "$budget" \
      --seed "$EVAL_SEED" \
      --device cuda:0
  ) >"$log" 2>&1
  echo "finished task=$task H=$horizon budget=$budget gpu=$gpu"
}

run_worker() {
  local task=$1
  local gpu=$2
  shift 2
  local horizon
  for horizon in "$@"; do
    run_cell "$task" "$horizon" "$gpu"
  done
}

tasks=(tworoom reacher pusht cube)
pids=()
for index in 0 1 2 3; do
  run_worker "${tasks[$index]}" "$index" 25 50 &
  pids+=("$!")
  run_worker "${tasks[$index]}" "$((index + 4))" 75 100 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

summary="$OUTPUT_ROOT/summary.tsv"
printf 'task\thorizon\teval_budget\tsuccess_rate\tlog\n' >"$summary"
for task in tworoom reacher pusht cube; do
  for horizon in 25 50 75 100; do
    budget=$((2 * horizon))
    log="$OUTPUT_ROOT/${task}_h${horizon}_b${budget}.log"
    rate=$(sed -n "s/.*'success_rate': \([0-9.]*\).*/\1/p" "$log" | tail -1)
    if [[ -z "$rate" ]]; then
      rate=ERROR
      failed=1
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$task" "$horizon" "$budget" "$rate" "$log" >>"$summary"
  done
done

cat "$summary"
echo "finished_at=$(date --iso-8601=seconds)"
if (( failed == 0 )); then
  touch "$OUTPUT_ROOT/DONE"
else
  touch "$OUTPUT_ROOT/FAILED"
fi
exit "$failed"
