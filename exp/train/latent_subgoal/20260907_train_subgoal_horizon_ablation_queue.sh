#!/usr/bin/env bash
set -euo pipefail

# Train the missing local-subgoal-horizon ablation generators sequentially on
# one GPU. QUEUE entries use FAMILY:SUBGOAL_STEPS:TASK, for example:
#   goalmax25:5:tworoom general_uniform_future:5:tworoom
# H25 uses goalmax25; H50 and longer horizons use general_uniform_future.

CLIENT_ID=${CLIENT_ID:-node3}
GPU_ID=${GPU_ID:?Set GPU_ID}
QUEUE=${QUEUE:?Set QUEUE to whitespace-separated FAMILY:K:TASK entries}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

PYTHON_BIN=${PYTHON_BIN_OVERRIDE:-$PYTHON_BIN}
RUNS_BASE=${RUNS_BASE:-/data-training/yyf/ogbench-lewm-policy-runs}
LEWM_LATENT_ROOT=${LEWM_LATENT_ROOT:-/data-training/yyf/datasets/lewm-latents}
TRAIN_STEPS=${TRAIN_STEPS:-200000}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
TRAIN_SEED=${TRAIN_SEED:-0}
ACTION_BLOCK=5
NUM_SAMPLES=8

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python executable: $PYTHON_BIN" >&2
  exit 2
fi

dataset_and_seed() {
  local task=$1
  case "$task" in
    tworoom)
      printf '%s|%s\n' "$LEWM_LATENT_ROOT/tworoom__lewm_s3072_e10_z192.h5" 3072
      ;;
    pusht)
      printf '%s|%s\n' "$LEWM_LATENT_ROOT/pusht_expert_train__lewm_s666_e10_z192.h5" 666
      ;;
    cube)
      printf '%s|%s\n' "$LEWM_LATENT_ROOT/cube_single_expert__lewm_s3072_e10_z192.h5" 3072
      ;;
    reacher)
      printf '%s|%s\n' "$LEWM_LATENT_ROOT/reacher__lewm_s3072_e10_z192.h5" 3072
      ;;
    *)
      echo "Unsupported task: $task" >&2
      return 2
      ;;
  esac
}

verify_config() {
  local config_path=$1 family=$2 subgoal_steps=$3
  "$PYTHON_BIN" - "$config_path" "$family" "$subgoal_steps" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
family = sys.argv[2]
subgoal_steps = int(sys.argv[3])
if not path.is_file():
    raise SystemExit(f'missing config: {path}')
config = json.loads(path.read_text())
expected = {
    'goalmax25': (
        'uniform_distance_first_aligned_future_same_trajectory_stride_5_max_25',
        25,
    ),
    'general_uniform_future': ('hiql_uniform_future_same_trajectory', None),
}[family]
actual = (
    config.get('goal_sampling'),
    config.get('max_goal_steps'),
)
if actual != expected:
    raise SystemExit(f'generator family mismatch at {path}: {actual!r} != {expected!r}')
if int(config.get('subgoal_steps', -1)) != subgoal_steps:
    raise SystemExit(f'wrong subgoal_steps at {path}')
if int(config.get('action_block', -1)) != 5:
    raise SystemExit(f'wrong action_block at {path}')
if int(config.get('flow_sampling_steps', -1)) != 16:
    raise SystemExit(f'wrong flow_sampling_steps at {path}')
print(f'verified family={family} k={subgoal_steps}: {path.parent.name}')
PY
}

run_one() {
  local family=$1 subgoal_steps=$2 task=$3
  local dataset_seed latent_dataset lewm_seed output_root exp_name run_dir checkpoint
  local -a goal_args

  if [[ "$subgoal_steps" != 5 && "$subgoal_steps" != 15 ]]; then
    echo "This launcher only accepts missing ablation values k=5 or k=15; got $subgoal_steps" >&2
    return 2
  fi

  dataset_seed=$(dataset_and_seed "$task")
  IFS='|' read -r latent_dataset lewm_seed <<< "$dataset_seed"
  if [[ ! -s "$latent_dataset" ]]; then
    echo "Missing latent dataset: $latent_dataset" >&2
    return 2
  fi

  case "$family" in
    goalmax25)
      output_root="$RUNS_BASE/latent-path-flow-k${subgoal_steps}-goalmax25"
      exp_name="latent_pathflow_${task}_lewm${lewm_seed}_hist3_sg${subgoal_steps}_ab5_goalstride5_goalmax25_cfm_ns8_n${TRAIN_STEPS}_b${TRAIN_BATCH_SIZE}_s${TRAIN_SEED}"
      goal_args=(--goal-sampling=aligned_future --max-goal-steps=25)
      ;;
    general_uniform_future)
      output_root="$RUNS_BASE/latent-path-flow-k${subgoal_steps}"
      exp_name="latent_pathflow_${task}_lewm${lewm_seed}_hist3_sg${subgoal_steps}_ab5_cfm_ns8_n${TRAIN_STEPS}_b${TRAIN_BATCH_SIZE}_s${TRAIN_SEED}"
      goal_args=(--goal-sampling=uniform_future)
      ;;
    *)
      echo "Unsupported family: $family" >&2
      return 2
      ;;
  esac

  run_dir="$output_root/$exp_name"
  checkpoint="$run_dir/checkpoint_${TRAIN_STEPS}.msgpack"
  if [[ -s "$checkpoint" ]]; then
    verify_config "$run_dir/config.json" "$family" "$subgoal_steps"
    echo "SKIP complete family=$family k=$subgoal_steps task=$task checkpoint=$checkpoint"
    return 0
  fi

  mkdir -p "$run_dir"
  echo "START $(date --iso-8601=seconds) family=$family k=$subgoal_steps task=$task gpu=$GPU_ID"
  (
    cd "$OGBENCH_ROOT/impls"
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_PLATFORMS=cuda \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" train_latent_subgoal_gcbc.py \
      --latent-dataset="$latent_dataset" \
      --save-dir="$run_dir" \
      --exp-name="$exp_name" \
      --architecture=latent_path_flow \
      --seed="$TRAIN_SEED" \
      --split-seed=0 \
      --train-fraction=0.95 \
      --subgoal-steps="$subgoal_steps" \
      --action-block="$ACTION_BLOCK" \
      "${goal_args[@]}" \
      --history-size=3 \
      --train-steps="$TRAIN_STEPS" \
      --batch-size="$TRAIN_BATCH_SIZE" \
      --hidden-dim=512 \
      --depth=4 \
      --num-heads=8 \
      --ff-dim=2048 \
      --time-dim=64 \
      --flow-sampling-steps=16 \
      --flow-solver=euler \
      --num-samples="$NUM_SAMPLES" \
      --ema-decay=0.9999 \
      --learning-rate=1e-4 \
      --final-learning-rate=1e-5 \
      --warmup-steps=5000 \
      --weight-decay=1e-4 \
      --gradient-clip=1.0 \
      --validation-pairs=10000 \
      --eval-batch-size=1024 \
      --log-interval=1000 \
      --eval-interval=10000 \
      --checkpoint-interval=25000 \
      --resume \
      2>&1 | tee -a "$run_dir/train.log"
  )

  if [[ ! -s "$checkpoint" ]]; then
    echo "Training exited without final checkpoint: $checkpoint" >&2
    return 1
  fi
  verify_config "$run_dir/config.json" "$family" "$subgoal_steps"
  echo "DONE $(date --iso-8601=seconds) family=$family k=$subgoal_steps task=$task checkpoint=$checkpoint"
}

echo "QUEUE_START $(date --iso-8601=seconds) client=$CLIENT_ID gpu=$GPU_ID queue=$QUEUE"
for entry in $QUEUE; do
  IFS=':' read -r family subgoal_steps task extra <<< "$entry"
  if [[ -n "${extra:-}" || -z "${family:-}" || -z "${subgoal_steps:-}" || -z "${task:-}" ]]; then
    echo "Malformed QUEUE entry: $entry" >&2
    exit 2
  fi
  run_one "$family" "$subgoal_steps" "$task"
done
echo "QUEUE_DONE $(date --iso-8601=seconds) client=$CLIENT_ID gpu=$GPU_ID"
