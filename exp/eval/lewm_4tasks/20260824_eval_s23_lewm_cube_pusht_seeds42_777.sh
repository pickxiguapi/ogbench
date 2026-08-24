#!/usr/bin/env bash
set -euo pipefail

# Server23：GPU2–5 并行验证 Cube/PushT × 训练 seed42/777 四个 LeWM-JAX epoch10；CEM300×30、H5/RH5、terminal、每项50 episodes。
CLIENT_ID=23
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/dzb/stablewm-data/lewm-jax-evals/20260824_cube_pusht_seeds42_777_official_cem300x30_h5_rh5_ep50_seed42}
EVAL_TMP_ROOT=${EVAL_TMP_ROOT:-/data/dzb/stablewm-data/tmp/lewm-cube-pusht-seeds42-777-eval}
source "$OGBENCH_ROOT/scripts/client_env.sh"
cd "$OGBENCH_ROOT/impls"

tasks=(cube pusht cube pusht)
tags=(cube_single pusht_expert cube_single pusht_expert)
seeds=(42 42 777 777)
gpus=(2 3 4 5)
pids=()

for i in "${!tasks[@]}"; do
  run_dir="/data/dzb/stablewm-data/lewm-jax-runs/LeWMJAX_impala_lance_${tags[$i]}_bs128_e10_seed${seeds[$i]}_fs5_h3_sigreg009_main20260823"
  output_dir="$OUTPUT_ROOT/${tasks[$i]}_seed${seeds[$i]}"
  task_tmp="$EVAL_TMP_ROOT/${tasks[$i]}_seed${seeds[$i]}"
  mkdir -p "$output_dir" "$task_tmp"
  TMPDIR="$task_tmp" CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" eval_lewm_4tasks.py \
    --task="${tasks[$i]}" --mode=lewm --data-root="$LEWM_DATA_ROOT" \
    --lewm-checkpoint="$run_dir/weights_epoch_10.msgpack" \
    --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" --goal-offset-steps=25 --eval-budget=50 \
    --cem-horizon=5 --cem-receding-horizon=5 --action-block=5 \
    --cem-num-samples=300 --cem-steps=30 --cem-topk=30 --cem-var-scale=1.0 \
    --cem-cost-mode=terminal --proposal-num-samples=64 --proposal-temperature=0.1 \
    --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
