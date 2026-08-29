#!/usr/bin/env bash
set -euo pipefail

# node4：评测 independent GCIQL-Chunk DDPG+BC 的三个 policy training seeds；每轮 8 卡并行运行四任务 Policy-only 与 Guided，统一使用 seed3072 LeWM、CEM300x5、H5/RH1、min-over-horizon、goal25/budget50。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
POLICY_STEPS=${POLICY_STEPS:-100000}
POLICY_BATCH_SIZE=${POLICY_BATCH_SIZE:-256}
P_AUG=${P_AUG:-0.5}
NUM_EVAL=${NUM_EVAL:-50}
GOAL_OFFSET_STEPS=${GOAL_OFFSET_STEPS:-25}
EVAL_BUDGET=${EVAL_BUDGET:-50}
CEM_HORIZON=${CEM_HORIZON:-5}
CEM_RECEDING_HORIZON=${CEM_RECEDING_HORIZON:-1}
CEM_NUM_SAMPLES=${CEM_NUM_SAMPLES:-300}
CEM_STEPS=${CEM_STEPS:-5}
CEM_TOPK=${CEM_TOPK:-30}
CEM_COST_MODE=${CEM_COST_MODE:-min_over_horizon}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-actor-ablation/evals/2026-08-29_ddpgbc_multiseed_policy_guided}
POLICY_ROOT=${POLICY_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-actor-ablation}
EVAL_TMP_ROOT=${EVAL_TMP_ROOT:-$POLICY_ROOT/tmp/ddpgbc-multiseed-policy-guided}
LEWM_ROOT=${LEWM_ROOT:-/data-training/yyf/models/lewm-jax-seed3072}
read -r -a policy_seeds <<< "${POLICY_SEEDS:-0 42 777}"
read -r -a eval_seeds <<< "${EVAL_SEEDS:-0 1 42}"
source "$OGBENCH_ROOT/scripts/client_env.sh"
cd "$OGBENCH_ROOT/impls"

tasks=(cube pusht reacher tworoom)
tags=(cube pusht reacher tworoom)
policy_gpus=(0 1 2 3)
guided_gpus=(4 5 6 7)
lewm_checkpoints=(
  "$LEWM_ROOT/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"
  "$LEWM_ROOT/LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"
  "$LEWM_ROOT/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"
  "$LEWM_ROOT/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"
)

for policy_seed in "${policy_seeds[@]}"; do
  for eval_seed in "${eval_seeds[@]}"; do
    pids=()
    for i in "${!tasks[@]}"; do
      policy_dir="$POLICY_ROOT/gc4_${tags[$i]}_ind_ddpgbc_alpha1_n${POLICY_STEPS}_b${POLICY_BATCH_SIZE}_a${P_AUG}_sd${policy_seed}"
      policy_output="$OUTPUT_ROOT/policy_seed${policy_seed}_eval${eval_seed}/${tags[$i]}"
      guided_output="$OUTPUT_ROOT/guided_seed${policy_seed}_eval${eval_seed}/${tags[$i]}"
      policy_tmp="$EVAL_TMP_ROOT/policy_seed${policy_seed}_eval${eval_seed}_${tags[$i]}"
      guided_tmp="$EVAL_TMP_ROOT/guided_seed${policy_seed}_eval${eval_seed}_${tags[$i]}"
      mkdir -p "$policy_output" "$guided_output" "$policy_tmp" "$guided_tmp"

      TMPDIR="$policy_tmp" CUDA_VISIBLE_DEVICES=${policy_gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
      LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
      "$PYTHON_BIN" eval_lewm_4tasks.py \
        --task="${tasks[$i]}" --mode=policy --data-root="$LEWM_DATA_ROOT" \
        --policy-checkpoint-dir="$policy_dir" --policy-checkpoint-step="$POLICY_STEPS" \
        --num-eval="$NUM_EVAL" --seed="$eval_seed" \
        --goal-offset-steps="$GOAL_OFFSET_STEPS" --eval-budget="$EVAL_BUDGET" \
        --output="$policy_output/result.json" >"$policy_output/eval.log" 2>&1 &
      pids+=("$!")

      TMPDIR="$guided_tmp" CUDA_VISIBLE_DEVICES=${guided_gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
      LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
      "$PYTHON_BIN" eval_lewm_4tasks.py \
        --task="${tasks[$i]}" --mode=guided --data-root="$LEWM_DATA_ROOT" \
        --lewm-checkpoint="${lewm_checkpoints[$i]}" \
        --policy-checkpoint-dir="$policy_dir" --policy-checkpoint-step="$POLICY_STEPS" \
        --num-eval="$NUM_EVAL" --seed="$eval_seed" \
        --goal-offset-steps="$GOAL_OFFSET_STEPS" --eval-budget="$EVAL_BUDGET" \
        --cem-horizon="$CEM_HORIZON" --cem-receding-horizon="$CEM_RECEDING_HORIZON" --action-block=5 \
        --cem-num-samples="$CEM_NUM_SAMPLES" --cem-steps="$CEM_STEPS" --cem-topk="$CEM_TOPK" \
        --cem-var-scale=1.0 --cem-cost-mode="$CEM_COST_MODE" \
        --output="$guided_output/result.json" >"$guided_output/eval.log" 2>&1 &
      pids+=("$!")
    done

    for pid in "${pids[@]}"; do
      wait "$pid"
    done
  done
done

"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import csv
import glob
import json
import os
import sys

root = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(root, '*_seed*_eval*', '*', 'result.json'))):
    with open(path) as file:
        result = json.load(file)
    group = os.path.basename(os.path.dirname(os.path.dirname(path)))
    mode, remainder = group.split('_seed', 1)
    policy_seed, eval_seed = remainder.split('_eval', 1)
    rows.append(
        [
            mode,
            int(policy_seed),
            int(eval_seed),
            result['task'],
            float(result['success_rate']),
            float(result['evaluation_time']),
            path,
        ]
    )

summary = os.path.join(root, 'summary.csv')
with open(summary, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['mode', 'policy_seed', 'eval_seed', 'task', 'success_rate', 'evaluation_time', 'result_path']
    )
    writer.writerows(rows)
print(summary)
PY
