#!/usr/bin/env bash
set -euo pipefail

# 四卡并行评测 LeWM-4Tasks；controller、subgoal 与 policy guidance 相互独立。
CLIENT_ID=${CLIENT_ID:-yb}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
CONTROLLER=${CONTROLLER:-lewm_cem}
POLICY_GUIDANCE=${POLICY_GUIDANCE:-mode}
USE_SUBGOAL=${USE_SUBGOAL:-0}
REPRESENTATION_MODE=${REPRESENTATION_MODE:-independent}
POLICY_STEPS=${POLICY_STEPS:-100000}
POLICY_SEED=${POLICY_SEED:-0}
POLICY_BATCH_SIZE=${POLICY_BATCH_SIZE:-256}
P_AUG=${P_AUG:-0.0}
LEWM_EPOCH=${LEWM_EPOCH:-10}
LEWM_SEED=${LEWM_SEED:-3072}
LEWM_BATCH_SIZE=${LEWM_BATCH_SIZE:-128}
NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
GOAL_OFFSET_STEPS=${GOAL_OFFSET_STEPS:-25}
EVAL_BUDGET=${EVAL_BUDGET:-50}
CEM_HORIZON=${CEM_HORIZON:-5}
CEM_RECEDING_HORIZON=${CEM_RECEDING_HORIZON:-1}
ACTION_BLOCK=${ACTION_BLOCK:-5}
CEM_NUM_SAMPLES=${CEM_NUM_SAMPLES:-300}
CEM_ITERATIONS=${CEM_ITERATIONS:-30}
CEM_TOPK=${CEM_TOPK:-30}
CEM_COST_MODE=${CEM_COST_MODE:-moh}
LATENT_SUBGOAL_STEPS=${LATENT_SUBGOAL_STEPS:-100000}
if [[ "$USE_SUBGOAL" == 1 ]]; then HORIZON_TAG=auto; else HORIZON_TAG=$CEM_HORIZON; fi
EVAL_TAG=${EVAL_TAG:-p${POLICY_STEPS}_w${LEWM_EPOCH}_cem${CEM_NUM_SAMPLES}x${CEM_ITERATIONS}_h${HORIZON_TAG}}
source "$OGBENCH_ROOT/scripts/client_env.sh"
EVAL_TMP_ROOT=${EVAL_TMP_ROOT:-$CLIENT_ROOT/tmp/lewm-4tasks-eval}
EGL_LIB_DIR=${EGL_LIB_DIR:-/usr/lib/x86_64-linux-gnu}
cd "$OGBENCH_ROOT/impls"

case "$CONTROLLER" in direct_policy|lewm_cem) ;; *) echo "CONTROLLER must be direct_policy or lewm_cem" >&2; exit 2 ;; esac
case "$POLICY_GUIDANCE" in none|mode) ;; *) echo "POLICY_GUIDANCE must be none or mode" >&2; exit 2 ;; esac
case "$USE_SUBGOAL" in 0|1) ;; *) echo "USE_SUBGOAL must be 0 or 1" >&2; exit 2 ;; esac
case "$REPRESENTATION_MODE" in independent|pi|qv|all) ;; *) echo "REPRESENTATION_MODE must be independent, pi, qv, or all" >&2; exit 2 ;; esac
if [[ "$CONTROLLER" == direct_policy && "$POLICY_GUIDANCE" != none ]]; then
  echo "direct_policy requires POLICY_GUIDANCE=none" >&2
  exit 2
fi
if [[ "$USE_SUBGOAL" == 1 && "$POLICY_GUIDANCE" != none && "$REPRESENTATION_MODE" != pi && "$REPRESENTATION_MODE" != all ]]; then
  echo "Subgoal-guided CEM requires REPRESENTATION_MODE=pi or all" >&2
  exit 2
fi
if [[ "$CONTROLLER" == direct_policy && "$USE_SUBGOAL" == 1 && "$REPRESENTATION_MODE" != pi && "$REPRESENTATION_MODE" != all ]]; then
  echo "Direct-policy latent subgoals require REPRESENTATION_MODE=pi or all" >&2
  exit 2
fi
case "$REPRESENTATION_MODE" in independent) MODE_TAG=ind ;; *) MODE_TAG=$REPRESENTATION_MODE ;; esac
tasks=(cube pusht reacher tworoom)
tags=(cube pusht reacher tworoom)
read -r -a gpus <<< "${GPU_IDS:-0 1 2 3}"
if (( ${#gpus[@]} != ${#tasks[@]} )); then
  echo "GPU_IDS must contain exactly four whitespace-separated GPU IDs." >&2
  exit 2
fi
default_lewm_root="$CLIENT_ROOT/lewm-final/lewm-4tasks"
lewm_dirs=(
  "${LEWM_CUBE_DIR:-$default_lewm_root/lewm_4tasks_cube_e${LEWM_EPOCH}_bs${LEWM_BATCH_SIZE}_s${LEWM_SEED}}"
  "${LEWM_PUSHT_DIR:-$default_lewm_root/lewm_4tasks_pusht_e${LEWM_EPOCH}_bs${LEWM_BATCH_SIZE}_s${LEWM_SEED}}"
  "${LEWM_REACHER_DIR:-$default_lewm_root/lewm_4tasks_reacher_e${LEWM_EPOCH}_bs${LEWM_BATCH_SIZE}_s${LEWM_SEED}}"
  "${LEWM_TWOROOM_DIR:-$default_lewm_root/lewm_4tasks_tworoom_e${LEWM_EPOCH}_bs${LEWM_BATCH_SIZE}_s${LEWM_SEED}}"
)
default_subgoal_root="$CLIENT_ROOT/lewm-final/latent-subgoal-gcbc-k10"
latent_subgoal_dirs=(
  "${LATENT_SUBGOAL_CUBE_DIR:-$default_subgoal_root/latent_gcbc_cube_lewm3072_k10_mse_n100000_b1024_s0}"
  "${LATENT_SUBGOAL_PUSHT_DIR:-$default_subgoal_root/latent_gcbc_pusht_lewm666_k10_mse_n100000_b1024_s0}"
  "${LATENT_SUBGOAL_REACHER_DIR:-$default_subgoal_root/latent_gcbc_reacher_lewm3072_k10_mse_n100000_b1024_s0}"
  "${LATENT_SUBGOAL_TWOROOM_DIR:-$default_subgoal_root/latent_gcbc_tworoom_lewm3072_k10_mse_n100000_b1024_s0}"
)
output_root=${OUTPUT_ROOT:-$CLIENT_ROOT/lewm-final/evals/lewm-4tasks/${CONTROLLER}_${POLICY_GUIDANCE}_sg${USE_SUBGOAL}_${REPRESENTATION_MODE}_${EVAL_TAG}_seed${EVAL_SEED}}
pids=()

for i in "${!tasks[@]}"; do
  lewm_dir=${lewm_dirs[$i]}
  latent_subgoal_dir=${latent_subgoal_dirs[$i]}
  policy_name="gc4_${tags[$i]}_${MODE_TAG}_n${POLICY_STEPS}_b${POLICY_BATCH_SIZE}_a${P_AUG}_sd${POLICY_SEED}"
  if (( ${#policy_name} >= 64 )); then
    echo "Policy experiment name must be shorter than 64 characters: $policy_name" >&2
    exit 2
  fi
  policy_dir="$CLIENT_ROOT/lewm-final/gciql-chunk-4tasks/$policy_name"
  args=()
  if [[ "$CONTROLLER" == lewm_cem ]]; then
    args+=(--lewm-checkpoint="$lewm_dir/weights_epoch_${LEWM_EPOCH}.msgpack")
  fi
  if [[ "$CONTROLLER" == direct_policy || "$POLICY_GUIDANCE" != none ]]; then
    args+=(--policy-checkpoint-dir="$policy_dir" --policy-checkpoint-step="$POLICY_STEPS")
  fi
  if [[ "$USE_SUBGOAL" == 1 ]]; then
    printf -v latent_subgoal_checkpoint "checkpoint_%06d.msgpack" "$LATENT_SUBGOAL_STEPS"
    args+=(
      --use-subgoal
      --latent-subgoal-checkpoint="$latent_subgoal_dir/$latent_subgoal_checkpoint"
    )
  fi
  output_dir="$output_root/${tags[$i]}"
  task_tmp="$EVAL_TMP_ROOT/${tags[$i]}"
  mkdir -p "$output_dir" "$task_tmp"
  TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" eval_lewm_4tasks.py \
    --task="${tasks[$i]}" --controller="$CONTROLLER" --policy-guidance="$POLICY_GUIDANCE" \
    --data-root="$LEWM_DATA_ROOT" "${args[@]}" \
    --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
    --goal-offset-steps="$GOAL_OFFSET_STEPS" --eval-budget="$EVAL_BUDGET" \
    --cem-horizon="$CEM_HORIZON" --cem-receding-horizon="$CEM_RECEDING_HORIZON" --action-block="$ACTION_BLOCK" \
    --cem-num-samples="$CEM_NUM_SAMPLES" --cem-iterations="$CEM_ITERATIONS" --cem-topk="$CEM_TOPK" --cem-var-scale=1.0 \
    --cem-cost-mode="$CEM_COST_MODE" \
    --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
