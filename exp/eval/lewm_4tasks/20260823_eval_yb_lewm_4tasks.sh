#!/usr/bin/env bash
set -euo pipefail

# 英博云：四卡并行评测 LeWM-4Tasks；MODE 可设 policy/lewm/guided/native_q，REPRESENTATION_MODE 可设 independent/pi/qv/all。
CLIENT_ID=yb
MODE=${MODE:-guided}
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
CEM_NUM_SAMPLES=${CEM_NUM_SAMPLES:-300}
CEM_STEPS=${CEM_STEPS:-5}
CEM_TOPK=${CEM_TOPK:-30}
PROPOSAL_NUM_SAMPLES=${PROPOSAL_NUM_SAMPLES:-64}
PROPOSAL_TEMPERATURE=${PROPOSAL_TEMPERATURE:-0.1}
EVAL_TAG=${EVAL_TAG:-p${POLICY_STEPS}_w${LEWM_EPOCH}_cem${CEM_NUM_SAMPLES}x${CEM_STEPS}_h${CEM_HORIZON}}
source /root/data/yyf/ogbench-new/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

case "$MODE" in policy|lewm|guided|native_q) ;; *) echo "MODE must be policy, lewm, guided, or native_q" >&2; exit 2 ;; esac
case "$REPRESENTATION_MODE" in independent|pi|qv|all) ;; *) echo "REPRESENTATION_MODE must be independent, pi, qv, or all" >&2; exit 2 ;; esac
tasks=(cube pusht reacher tworoom)
tags=(cube pusht reacher tworoom)
gpus=(0 1 2 3)
output_root="$CLIENT_ROOT/lewm-final/evals/lewm-4tasks/${MODE}_${REPRESENTATION_MODE}_${EVAL_TAG}_seed${EVAL_SEED}"
pids=()

for i in "${!tasks[@]}"; do
  lewm_dir="$CLIENT_ROOT/lewm-final/lewm-4tasks/lewm_4tasks_${tags[$i]}_e${LEWM_EPOCH}_bs${LEWM_BATCH_SIZE}_s${LEWM_SEED}"
  policy_dir="$CLIENT_ROOT/lewm-final/gciql-chunk-4tasks/gciql_chunk_4tasks_${tags[$i]}_${REPRESENTATION_MODE}_s${POLICY_STEPS}_bs${POLICY_BATCH_SIZE}_paug${P_AUG}_s${POLICY_SEED}"
  args=()
  [[ "$MODE" != policy ]] && args+=(--lewm-checkpoint="$lewm_dir/weights_epoch_${LEWM_EPOCH}.msgpack")
  [[ "$MODE" != lewm ]] && args+=(--policy-checkpoint-dir="$policy_dir" --policy-checkpoint-step="$POLICY_STEPS")
  output_dir="$output_root/${tags[$i]}"
  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" eval_lewm_4tasks.py \
    --task="${tasks[$i]}" --mode="$MODE" --data-root="$LEWM_DATA_ROOT" "${args[@]}" \
    --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
    --goal-offset-steps="$GOAL_OFFSET_STEPS" --eval-budget="$EVAL_BUDGET" \
    --cem-horizon="$CEM_HORIZON" --cem-receding-horizon=1 --action-block=5 \
    --cem-num-samples="$CEM_NUM_SAMPLES" --cem-steps="$CEM_STEPS" --cem-topk="$CEM_TOPK" --cem-var-scale=1.0 \
    --proposal-num-samples="$PROPOSAL_NUM_SAMPLES" --proposal-temperature="$PROPOSAL_TEMPERATURE" \
    --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
