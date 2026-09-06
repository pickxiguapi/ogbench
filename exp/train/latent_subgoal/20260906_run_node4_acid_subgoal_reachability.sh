#!/usr/bin/env bash
set -euo pipefail

# A800 node4: reproduce ACID's independent flow-matching IDM for the mixed
# seed666/3072 LeWM encoders, then rerun the canonical H50 LeWM++ predictor
# ablation with exact traces and measure ACID consistency plus real Hit@K.
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

MODE=${MODE:-launch}
SESSION=${SESSION:-acid-h50-reachability}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
TRAIN_STEPS=${TRAIN_STEPS:-200000}
NUM_EVAL=${NUM_EVAL:-50}
ARCHITECTURES=${ARCHITECTURES:-"history_mlp endpoint_flow latent_path_flow"}
TRAIN_SEEDS=${TRAIN_SEEDS:-"0 1 42"}
EVAL_SEEDS=${EVAL_SEEDS:-"0 1 42"}
POLICY_SEED=${POLICY_SEED:-777}
POLICY_STEPS=100000

LATENT_ROOT=${LATENT_ROOT:-/data-training/yyf/datasets/lewm-latents}
IDM_ROOT=${IDM_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/acid-idm-k5-lewm-mixed666-3072}
SOURCE_PREDICTOR_ROOT=${SOURCE_PREDICTOR_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-predictor-h50-ablation}
GENERAL_ROOT=${GENERAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/latent-path-flow-k10}
PREDICTOR_VIEW_ROOT=${PREDICTOR_VIEW_ROOT:-$GENERAL_ROOT/general_uniform_future_h50_predictor_ablation}
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-node3-mirror}
EVAL_ROOT=${EVAL_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks/20260906_acid_reachability_h50_general_uniform_future_lewmpp_policy777_ns1_cem300x5_h2_rh1_g50_b100_ep${NUM_EVAL}}
TMP_ROOT=${TMP_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/tmp/20260906-acid-reachability}
DRIVER_LOG=${DRIVER_LOG:-$EVAL_ROOT/driver.log}

tasks=(cube pusht reacher tworoom)
lewm_seeds=(3072 666 3072 3072)
latent_datasets=(
  "$LATENT_ROOT/cube_single_expert__lewm_s3072_e10_z192.h5"
  "$LATENT_ROOT/pusht_expert_train__lewm_s666_e10_z192.h5"
  "$LATENT_ROOT/reacher__lewm_s3072_e10_z192.h5"
  "$LATENT_ROOT/tworoom__lewm_s3072_e10_z192.h5"
)
lewm_checkpoints=(
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
)

read -r -a gpu_ids <<<"$GPU_IDS"
read -r -a architectures <<<"$ARCHITECTURES"
read -r -a train_seeds <<<"$TRAIN_SEEDS"
read -r -a eval_seeds <<<"$EVAL_SEEDS"
if (( ${#gpu_ids[@]} != 8 )); then
  echo "GPU_IDS must contain exactly eight GPU IDs." >&2
  exit 2
fi

idm_dir() {
  local task=$1 lewm_seed=$2
  echo "$IDM_ROOT/acid_idm_${task}_lewm${lewm_seed}_k5_flow4x192h3_n${TRAIN_STEPS}_b256_s0"
}

verify_generator() {
  local checkpoint=$1 architecture=$2 train_seed=$3
  "$PYTHON_BIN" - "$checkpoint" "$architecture" "$train_seed" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve()
requested = sys.argv[2]
train_seed = int(sys.argv[3])
expected_architecture = {
    'history_mlp': 'history_latent_mlp',
    'endpoint_flow': 'latent_endpoint_flow_transformer_encoder',
    'latent_path_flow': 'latent_path_flow_transformer_encoder',
}[requested]
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
config_path = checkpoint.parent / 'config.json'
config = json.loads(config_path.read_text())
expected = {
    'architecture': expected_architecture,
    'goal_sampling': 'hiql_uniform_future_same_trajectory',
    'max_goal_steps': None,
    'subgoal_steps': 10,
    'action_block': 5,
    'history_size': 3,
    'seed': train_seed,
    'train_steps': 200000,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise ValueError(f'{config_path}: {key}={config.get(key)!r}, expected {value!r}')
print(f'verified general_uniform_future generator: {checkpoint}')
PY
}

stage_predictor_view() {
  mkdir -p "$PREDICTOR_VIEW_ROOT"
  local architecture train_seed i task lewm_seed exp_name source_dir view_dir
  for architecture in "${architectures[@]}"; do
    for train_seed in "${train_seeds[@]}"; do
      for i in "${!tasks[@]}"; do
        task=${tasks[$i]}
        lewm_seed=${lewm_seeds[$i]}
        exp_name="h50_${architecture}_${task}_lewm${lewm_seed}_hist3_k10_pmatch18m_n200000_b1024_s${train_seed}"
        source_dir="$SOURCE_PREDICTOR_ROOT/$exp_name"
        view_dir="$PREDICTOR_VIEW_ROOT/$exp_name"
        verify_generator "$source_dir/checkpoint_200000.msgpack" "$architecture" "$train_seed"
        if [[ -L "$view_dir" ]]; then
          [[ $(readlink -f "$view_dir") == $(readlink -f "$source_dir") ]] || {
            echo "Predictor view points elsewhere: $view_dir" >&2
            exit 3
          }
        elif [[ -e "$view_dir" ]]; then
          echo "Predictor view exists but is not a symlink: $view_dir" >&2
          exit 3
        else
          ln -s "$source_dir" "$view_dir"
        fi
      done
    done
  done
}

train_idm_task() {
  local task=$1 lewm_seed=$2 latent_dataset=$3 gpu=$4
  local output_dir
  output_dir=$(idm_dir "$task" "$lewm_seed")
  mkdir -p "$output_dir" "$TMP_ROOT/idm/$task"
  (
    cd "$OGBENCH_ROOT/impls"
    TMPDIR="$TMP_ROOT/idm/$task" CUDA_VISIBLE_DEVICES="$gpu" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" train_acid_idm.py \
      --latent-dataset="$latent_dataset" \
      --save-dir="$output_dir" \
      --exp-name="$(basename "$output_dir")" \
      --seed=0 --split-seed=0 --train-fraction=0.9 \
      --transition-steps=5 --train-steps="$TRAIN_STEPS" --batch-size=256 \
      --model-dim=192 --num-layers=4 --num-heads=3 --mlp-dim=768 \
      --learning-rate=1e-4 --final-learning-rate=1e-6 \
      --warmup-steps=2000 --weight-decay=1e-4 \
      --validation-pairs=50000 --eval-batch-size=5000 \
      --log-interval=1000 --eval-interval=5000 \
      --checkpoint-interval=25000 --resume \
      >"$output_dir/train.log" 2>&1
  )
}

run_setting() {
  local gpu_offset=$1 architecture=$2 train_seed=$3 eval_seed=$4
  local -a pids=()
  local i task lewm_seed lewm_checkpoint exp_name checkpoint policy_dir output_dir trace_dir idm_checkpoint task_tmp
  for i in "${!tasks[@]}"; do
    task=${tasks[$i]}
    lewm_seed=${lewm_seeds[$i]}
    lewm_checkpoint=${lewm_checkpoints[$i]}
    exp_name="h50_${architecture}_${task}_lewm${lewm_seed}_hist3_k10_pmatch18m_n200000_b1024_s${train_seed}"
    checkpoint="$PREDICTOR_VIEW_ROOT/$exp_name/checkpoint_200000.msgpack"
    verify_generator "$checkpoint" "$architecture" "$train_seed"
    policy_dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd${POLICY_SEED}"
    output_dir="$EVAL_ROOT/$architecture/train${train_seed}/eval${eval_seed}/$task"
    trace_dir="$output_dir/trace"
    idm_checkpoint="$(idm_dir "$task" "$lewm_seed")/checkpoint_$(printf '%06d' "$TRAIN_STEPS").msgpack"
    task_tmp="$TMP_ROOT/eval/$architecture/train${train_seed}/eval${eval_seed}/$task"
    mkdir -p "$output_dir" "$trace_dir" "$task_tmp"
    if [[ -s "$output_dir/result.json" && -s "$output_dir/reachability.json" ]]; then
      echo "skip complete $architecture train=$train_seed eval=$eval_seed task=$task"
      continue
    fi
    (
      cd "$OGBENCH_ROOT/impls"
      TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES="${gpu_ids[$((gpu_offset + i))]}" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
      LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
      "$PYTHON_BIN" eval_lewm_4tasks.py \
        --task="$task" --controller=lewm_cem --policy-guidance=mode \
        --use-subgoal --guidance-goal-mode=final \
        --guidance-population-size=0 --guidance-temperature=1.0 \
        --guidance-elite-size=8 \
        --data-root="$LEWM_DATA_ROOT" --lewm-checkpoint="$lewm_checkpoint" \
        --policy-checkpoint-dir="$policy_dir" --policy-checkpoint-step="$POLICY_STEPS" \
        --latent-subgoal-checkpoint="$checkpoint" --num-samples=1 \
        --num-eval="$NUM_EVAL" --seed="$eval_seed" \
        --goal-offset-steps=50 --eval-budget=100 \
        --cem-horizon=2 --cem-receding-horizon=1 --action-block=5 \
        --cem-num-samples=300 --cem-iterations=5 --cem-topk=30 \
        --cem-var-scale=1.0 --cem-cost-mode=moh \
        --trace-dir="$trace_dir" --output="$output_dir/result.json" \
        >"$output_dir/eval.log" 2>&1
      CUDA_VISIBLE_DEVICES="${gpu_ids[$((gpu_offset + i))]}" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
      "$PYTHON_BIN" score_acid_subgoal_reachability.py \
        --task="$task" --trace-dir="$trace_dir" \
        --lewm-checkpoint="$lewm_checkpoint" --idm-checkpoint="$idm_checkpoint" \
        --real-horizon=10 --transition-steps=5 --seed=0 \
        --output="$output_dir/reachability.json" \
        >"$output_dir/reachability.log" 2>&1
    ) &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  return "$failed"
}

driver() {
  mkdir -p "$IDM_ROOT" "$EVAL_ROOT" "$TMP_ROOT"
  stage_predictor_view
  local i used failed=0
  local -a pids=()
  for i in "${!tasks[@]}"; do
    train_idm_task "${tasks[$i]}" "${lewm_seeds[$i]}" \
      "${latent_datasets[$i]}" "${gpu_ids[$i]}" &
    pids+=("$!")
  done
  for used in "${pids[@]}"; do
    if ! wait "$used"; then failed=1; fi
  done
  (( failed == 0 )) || { echo "At least one IDM training failed." >&2; exit 1; }
  for i in "${!tasks[@]}"; do
    test -s "$(idm_dir "${tasks[$i]}" "${lewm_seeds[$i]}")/checkpoint_$(printf '%06d' "$TRAIN_STEPS").msgpack"
  done

  local -a setting_architectures=() setting_train_seeds=() setting_eval_seeds=()
  local architecture train_seed eval_seed base slot index
  for architecture in "${architectures[@]}"; do
    for train_seed in "${train_seeds[@]}"; do
      for eval_seed in "${eval_seeds[@]}"; do
        setting_architectures+=("$architecture")
        setting_train_seeds+=("$train_seed")
        setting_eval_seeds+=("$eval_seed")
      done
    done
  done
  for (( base=0; base<${#setting_architectures[@]}; base+=2 )); do
    pids=()
    for slot in 0 1; do
      index=$((base + slot))
      (( index < ${#setting_architectures[@]} )) || continue
      run_setting "$((slot * 4))" "${setting_architectures[$index]}" \
        "${setting_train_seeds[$index]}" "${setting_eval_seeds[$index]}" &
      pids+=("$!")
    done
    for used in "${pids[@]}"; do
      if ! wait "$used"; then failed=1; fi
    done
    (( failed == 0 )) || { echo "Reachability evaluation failed." >&2; exit 1; }
  done
  "$PYTHON_BIN" "$OGBENCH_ROOT/impls/aggregate_acid_subgoal_reachability.py" \
    --root="$EVAL_ROOT" --output="$EVAL_ROOT/aggregate.json"
  echo "DONE: $EVAL_ROOT"
}

case "$MODE" in
  launch)
    mkdir -p "$EVAL_ROOT"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "tmux session already exists: $SESSION" >&2
      exit 3
    fi
    printf -v command '%q ' env MODE=driver SESSION="$SESSION" GPU_IDS="$GPU_IDS" \
      TRAIN_STEPS="$TRAIN_STEPS" NUM_EVAL="$NUM_EVAL" \
      ARCHITECTURES="$ARCHITECTURES" TRAIN_SEEDS="$TRAIN_SEEDS" EVAL_SEEDS="$EVAL_SEEDS" \
      POLICY_SEED="$POLICY_SEED" LATENT_ROOT="$LATENT_ROOT" IDM_ROOT="$IDM_ROOT" \
      SOURCE_PREDICTOR_ROOT="$SOURCE_PREDICTOR_ROOT" GENERAL_ROOT="$GENERAL_ROOT" \
      PREDICTOR_VIEW_ROOT="$PREDICTOR_VIEW_ROOT" POLICY_ROOT="$POLICY_ROOT" \
      EVAL_ROOT="$EVAL_ROOT" TMP_ROOT="$TMP_ROOT" \
      bash exp/train/latent_subgoal/20260906_run_node4_acid_subgoal_reachability.sh
    tmux new-session -d -s "$SESSION" -c "$OGBENCH_ROOT" \
      "bash -lc '$command >\"$DRIVER_LOG\" 2>&1'"
    echo "launched tmux=$SESSION output=$EVAL_ROOT log=$DRIVER_LOG"
    ;;
  driver)
    driver
    ;;
  status)
    tmux list-sessions 2>/dev/null | grep "$SESSION" || true
    echo "===== driver ====="
    tail -n 30 "$DRIVER_LOG" 2>/dev/null || true
    for i in "${!tasks[@]}"; do
      echo "===== IDM ${tasks[$i]} ====="
      tail -n 8 "$(idm_dir "${tasks[$i]}" "${lewm_seeds[$i]}")/train.log" 2>/dev/null || true
    done
    ;;
  *)
    echo "MODE must be launch, driver, or status." >&2
    exit 2
    ;;
esac
