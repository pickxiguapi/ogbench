#!/usr/bin/env bash
set -euo pipefail

# Yingbo: evaluate the completed shared-all GCIQL-Chunk-AWR policy seed132 on
# Cube, PushT, and Reacher. Policy-only and CEM+policy run concurrently on two
# three-GPU groups. The guided planner must use the same frozen LeWM checkpoint
# as the shared-all policy for each task.
CLIENT_ID=yb
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

POLICY_SEED=${POLICY_SEED:-132}
POLICY_STEPS=${POLICY_STEPS:-100000}
POLICY_BATCH_SIZE=${POLICY_BATCH_SIZE:-256}
P_AUG=${P_AUG:-0.0}
NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
GOAL_OFFSET_STEPS=${GOAL_OFFSET_STEPS:-25}
EVAL_BUDGET=${EVAL_BUDGET:-50}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/data/yyf/lewm-final/evals/lewm-4tasks/20260828_gciql_chunk_all_seed132_3tasks}
EVAL_TMP_ROOT=${EVAL_TMP_ROOT:-/root/data/yyf/tmp/lewm-4tasks-eval-all-seed132}
EGL_LIB_DIR=${EGL_LIB_DIR:-/root/data/yyf/egl-runtime/root/usr/lib/x86_64-linux-gnu}

CEM_HORIZON=${CEM_HORIZON:-5}
CEM_RECEDING_HORIZON=${CEM_RECEDING_HORIZON:-1}
CEM_NUM_SAMPLES=${CEM_NUM_SAMPLES:-300}
CEM_STEPS=${CEM_STEPS:-5}
CEM_TOPK=${CEM_TOPK:-30}
CEM_COST_MODE=${CEM_COST_MODE:-min_over_horizon}

source "$OGBENCH_ROOT/scripts/client_env.sh"
cd "$OGBENCH_ROOT/impls"

tasks=(cube pusht reacher)
tags=(cube pusht reacher)
read -r -a policy_gpus <<< "${POLICY_GPU_IDS:-0 1 2}"
read -r -a guided_gpus <<< "${GUIDED_GPU_IDS:-5 6 7}"
if (( ${#policy_gpus[@]} != 3 || ${#guided_gpus[@]} != 3 )); then
  echo "POLICY_GPU_IDS and GUIDED_GPU_IDS must each contain exactly three GPU IDs." >&2
  exit 2
fi

lewm_seed3072_root=/root/data/yyf/lewm-jax-seed3072-s23
lewm_seed666_root=/root/data/yyf/lewm-jax-seed666-s23
lewm_checkpoints=(
  "$lewm_seed3072_root/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"
  "$lewm_seed666_root/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack"
  "$lewm_seed3072_root/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"
)

policy_dirs=()
for tag in "${tags[@]}"; do
  policy_dirs+=("$CLIENT_ROOT/lewm-final/gciql-chunk-4tasks/gc4_${tag}_all_n${POLICY_STEPS}_b${POLICY_BATCH_SIZE}_a${P_AUG}_sd${POLICY_SEED}")
done

for i in "${!tasks[@]}"; do
  [[ -s "${lewm_checkpoints[$i]}" ]] || {
    echo "Missing LeWM checkpoint: ${lewm_checkpoints[$i]}" >&2
    exit 2
  }
  [[ -s "${policy_dirs[$i]}/params_${POLICY_STEPS}.pkl" ]] || {
    echo "Missing policy checkpoint: ${policy_dirs[$i]}/params_${POLICY_STEPS}.pkl" >&2
    exit 2
  }
done

run_eval() {
  local mode=$1
  local task=$2
  local gpu=$3
  local lewm_checkpoint=$4
  local policy_dir=$5
  local output_dir="$OUTPUT_ROOT/${mode}_seed${POLICY_SEED}/${task}"
  local task_tmp="$EVAL_TMP_ROOT/${mode}_${task}"
  local args=(
    --task="$task"
    --mode="$mode"
    --data-root="$LEWM_DATA_ROOT"
    --policy-checkpoint-dir="$policy_dir"
    --policy-checkpoint-step="$POLICY_STEPS"
    --num-eval="$NUM_EVAL"
    --seed="$EVAL_SEED"
    --goal-offset-steps="$GOAL_OFFSET_STEPS"
    --eval-budget="$EVAL_BUDGET"
    --cem-horizon="$CEM_HORIZON"
    --cem-receding-horizon="$CEM_RECEDING_HORIZON"
    --action-block=5
    --cem-num-samples="$CEM_NUM_SAMPLES"
    --cem-steps="$CEM_STEPS"
    --cem-topk="$CEM_TOPK"
    --cem-var-scale=1.0
    --cem-cost-mode="$CEM_COST_MODE"
    --output="$output_dir/result.json"
  )
  if [[ "$mode" == guided ]]; then
    args+=(--lewm-checkpoint="$lewm_checkpoint")
  fi

  mkdir -p "$output_dir" "$task_tmp"
  TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
    LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" eval_lewm_4tasks.py "${args[@]}" >"$output_dir/eval.log" 2>&1
}

pids=()
for i in "${!tasks[@]}"; do
  run_eval policy "${tasks[$i]}" "${policy_gpus[$i]}" \
    "${lewm_checkpoints[$i]}" "${policy_dirs[$i]}" &
  pids+=("$!")
  run_eval guided "${tasks[$i]}" "${guided_gpus[$i]}" \
    "${lewm_checkpoints[$i]}" "${policy_dirs[$i]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
if (( status != 0 )); then
  echo "At least one evaluation failed; inspect logs under $OUTPUT_ROOT." >&2
  exit "$status"
fi

echo "All shared-all seed132 evaluations completed: $OUTPUT_ROOT"
