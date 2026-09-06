#!/usr/bin/env bash
set -euo pipefail

# Compare subgoal generators on identical dataset states. Each generator sees
# the same observation/goal pairs and uses paired CEM keys; only its selected
# two-block plan is replayed in the simulator and scored by ACID.
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$OGBENCH_ROOT/scripts/client_env.sh"

MODE=${MODE:-launch}
SESSION=${SESSION:-acid-paired-generators-h50-states10}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
NUM_STATES=${NUM_STATES:-10}
ARCHITECTURES=${ARCHITECTURES:-"history_mlp endpoint_flow latent_path_flow"}
TRAIN_SEED=${TRAIN_SEED:-0}
EVAL_SEED=${EVAL_SEED:-42}
ROOT=/data-training/yyf/ogbench-lewm-policy-runs
PREDICTOR_ROOT=${PREDICTOR_ROOT:-$ROOT/latent-predictor-h50-ablation}
POLICY_ROOT=${POLICY_ROOT:-$ROOT/gciql-chunk-4tasks-node3-mirror}
IDM_ROOT=${IDM_ROOT:-$ROOT/acid-idm-k5-lewm-mixed666-3072}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/evals/lewm-4tasks/20260906_acid_paired_subgoal_generators_h50_general_uniform_future_lewmpp_policy777_ns1_cem300x5_h2_states${NUM_STATES}}
TMP_ROOT=${TMP_ROOT:-$ROOT/tmp/20260906-acid-paired-subgoal-generators}
DRIVER_LOG=${DRIVER_LOG:-$OUTPUT_ROOT/driver.log}

tasks=(cube pusht reacher tworoom)
lewm_seeds=(3072 666 3072 3072)
lewm_checkpoints=(
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed666/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
  /data-training/yyf/models/lewm-jax-seed3072/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack
)
read -r -a gpu_ids <<<"$GPU_IDS"
read -r -a architectures <<<"$ARCHITECTURES"
(( ${#gpu_ids[@]} == 8 )) || { echo "GPU_IDS must contain eight IDs" >&2; exit 2; }

run_architecture() {
  local gpu_offset=$1 architecture=$2 i task lewm_seed exp_name checkpoint out
  local -a pids=()
  for i in "${!tasks[@]}"; do
    task=${tasks[$i]}; lewm_seed=${lewm_seeds[$i]}
    exp_name="h50_${architecture}_${task}_lewm${lewm_seed}_hist3_k10_pmatch18m_n200000_b1024_s${TRAIN_SEED}"
    checkpoint="$PREDICTOR_ROOT/$exp_name/checkpoint_200000.msgpack"
    out="$OUTPUT_ROOT/$architecture/$task"
    mkdir -p "$out" "$TMP_ROOT/$architecture/$task"
    if [[ -s "$out/paired.json" ]]; then echo "skip $architecture/$task"; continue; fi
    (
      cd "$OGBENCH_ROOT/impls"
      TMPDIR="$TMP_ROOT/$architecture/$task" \
      CUDA_VISIBLE_DEVICES="${gpu_ids[$((gpu_offset+i))]}" \
      XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
      LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
      "$PYTHON_BIN" eval_acid_fixed_candidates.py \
        --task="$task" --data-root="$LEWM_DATA_ROOT" \
        --lewm-checkpoint="${lewm_checkpoints[$i]}" \
        --idm-checkpoint="$IDM_ROOT/acid_idm_${task}_lewm${lewm_seed}_k5_flow4x192h3_n200000_b256_s0/checkpoint_200000.msgpack" \
        --policy-checkpoint-dir="$POLICY_ROOT/gc4_${task}_all_n100000_b256_a0.0_sd777" \
        --policy-checkpoint-step=100000 --latent-subgoal-checkpoint="$checkpoint" \
        --num-subgoal-samples=1 --replay-scope=selected_plan \
        --num-states="$NUM_STATES" --seed="$EVAL_SEED" --goal-offset-steps=50 \
        --action-block=5 --cem-horizon=2 --cem-receding-horizon=1 \
        --cem-num-samples=300 --cem-iterations=5 --cem-topk=30 \
        --cem-var-scale=1.0 --cem-cost-mode=moh --output="$out/paired.json" \
        >"$out/paired.log" 2>&1
    ) & pids+=("$!")
  done
  local pid failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return "$failed"
}

driver() {
  mkdir -p "$OUTPUT_ROOT" "$TMP_ROOT"
  local base slot index pid failed=0
  local -a pids=()
  for ((base=0; base<${#architectures[@]}; base+=2)); do
    pids=()
    for slot in 0 1; do
      index=$((base+slot)); ((index<${#architectures[@]})) || continue
      run_architecture "$((slot*4))" "${architectures[$index]}" & pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    ((failed==0)) || { echo "paired generator evaluation failed" >&2; exit 1; }
  done
  "$PYTHON_BIN" "$OGBENCH_ROOT/impls/aggregate_acid_generator_feasibility.py" \
    --root="$OUTPUT_ROOT" --output="$OUTPUT_ROOT/aggregate.json" \
    --architectures "${architectures[@]}" --tasks "${tasks[@]}"
  echo "DONE: $OUTPUT_ROOT"
}

case "$MODE" in
  launch)
    mkdir -p "$OUTPUT_ROOT"
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session exists: $SESSION" >&2; exit 3; }
    printf -v command '%q ' env MODE=driver SESSION="$SESSION" GPU_IDS="$GPU_IDS" \
      NUM_STATES="$NUM_STATES" ARCHITECTURES="$ARCHITECTURES" TRAIN_SEED="$TRAIN_SEED" \
      EVAL_SEED="$EVAL_SEED" OUTPUT_ROOT="$OUTPUT_ROOT" TMP_ROOT="$TMP_ROOT" \
      bash exp/train/latent_subgoal/20260906_run_node4_acid_paired_generator_feasibility.sh
    printf -v quoted_log '%q' "$DRIVER_LOG"
    tmux new-session -d -s "$SESSION" -c "$OGBENCH_ROOT" "$command >$quoted_log 2>&1"
    echo "launched tmux=$SESSION log=$DRIVER_LOG"
    ;;
  driver) driver ;;
  status) tmux ls 2>/dev/null | grep "$SESSION" || true; tail -n 30 "$DRIVER_LOG" 2>/dev/null || true ;;
  *) echo "MODE must be launch, driver, or status" >&2; exit 2 ;;
esac
