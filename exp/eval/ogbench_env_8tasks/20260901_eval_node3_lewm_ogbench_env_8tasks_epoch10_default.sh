#!/usr/bin/env bash
set -euo pipefail

# A800 node3：八卡并行评测 OGBench-Env-8Tasks 的 LeWM-JAX epoch10；纯 LeWM 默认 planning（H5/RH1、CEM300x30、topk30、MoH），每个内部 task 50 episodes、eval seed42。
CLIENT_ID=node3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
LEWM_EPOCH=${LEWM_EPOCH:-10}
LEWM_EPOCHS=${LEWM_EPOCHS:-10}
LEWM_BATCH_SIZE=${LEWM_BATCH_SIZE:-128}
LEWM_SEED=${LEWM_SEED:-3072}
NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
CEM_HORIZON=${CEM_HORIZON:-5}
CEM_RECEDING_HORIZON=${CEM_RECEDING_HORIZON:-1}
ACTION_BLOCK=${ACTION_BLOCK:-5}
CEM_NUM_SAMPLES=${CEM_NUM_SAMPLES:-300}
CEM_ITERATIONS=${CEM_ITERATIONS:-30}
CEM_TOPK=${CEM_TOPK:-30}
CEM_VAR_SCALE=${CEM_VAR_SCALE:-1.0}
CEM_COST_MODE=${CEM_COST_MODE:-moh}
source "$OGBENCH_ROOT/scripts/client_env.sh"
LEWM_RUN_ROOT=${LEWM_RUN_ROOT:-$CLIENT_ROOT/ogbench-lewm-policy-runs/lewm-ogbench8}
EVAL_ROOT=${EVAL_ROOT:-$CLIENT_ROOT/ogbench-lewm-policy-runs/evals/ogbench-env-8tasks}
cd "$OGBENCH_ROOT/impls"

envs=(visual-cube-single-play-v0 visual-cube-double-play-v0 visual-cube-triple-play-v0 visual-scene-play-v0 visual-cube-single-noisy-v0 visual-cube-double-noisy-v0 visual-cube-triple-noisy-v0 visual-scene-noisy-v0)
tags=(cs_play cd_play ct_play scene_play cs_noisy cd_noisy ct_noisy scene_noisy)
output_root="$EVAL_ROOT/lewm_cem_none_epoch${LEWM_EPOCH}_cem${CEM_NUM_SAMPLES}x${CEM_ITERATIONS}_h${CEM_HORIZON}_rh${CEM_RECEDING_HORIZON}_${CEM_COST_MODE}_seed${EVAL_SEED}"
pids=()

for i in "${!envs[@]}"; do
  lewm_dir="$LEWM_RUN_ROOT/lewm_ogbench8_${tags[$i]}_e${LEWM_EPOCHS}_bs${LEWM_BATCH_SIZE}_s${LEWM_SEED}"
  output_dir="$output_root/${tags[$i]}"
  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES=$i XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" eval_ogbench_env_8tasks.py \
    --env-name="${envs[$i]}" --dataset-path="$OGBENCH_DATA_DIR/${envs[$i]}.npz" \
    --controller=lewm_cem --policy-guidance=none \
    --lewm-checkpoint="$lewm_dir/weights_epoch_${LEWM_EPOCH}.msgpack" \
    --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
    --cem-horizon="$CEM_HORIZON" --cem-receding-horizon="$CEM_RECEDING_HORIZON" \
    --action-block="$ACTION_BLOCK" --cem-num-samples="$CEM_NUM_SAMPLES" \
    --cem-iterations="$CEM_ITERATIONS" --cem-topk="$CEM_TOPK" \
    --cem-var-scale="$CEM_VAR_SCALE" --cem-cost-mode="$CEM_COST_MODE" \
    --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
