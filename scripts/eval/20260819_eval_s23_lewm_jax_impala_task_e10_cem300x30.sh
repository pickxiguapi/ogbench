#!/usr/bin/env bash
set -euo pipefail

# Server 23：评测 LeWM-JAX IMPALA epoch-10；单任务 50 episodes，dataset-goal，CEM 300 samples × 30 iterations。
task="${1:-}"
CLIENT_ID=23
source /home/dzb/ogbench/scripts/client_env.sh

case "$task" in
  cube)
    exp_name=LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
    gpu=2
    ;;
  pusht)
    exp_name=LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
    gpu=3
    ;;
  reacher)
    exp_name=LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
    gpu=4
    ;;
  tworoom)
    exp_name=LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
    gpu=5
    ;;
  *)
    echo "Usage: bash scripts/eval/20260819_eval_s23_lewm_jax_impala_task_e10_cem300x30.sh {cube|pusht|reacher|tworoom}" >&2
    exit 2
    ;;
esac

run_dir="/data/dzb/stablewm-data/lewm-jax-runs/$exp_name"
output_dir="$run_dir/eval_cem300x30_seed42_20260819"
mkdir -p "$output_dir/videos"
cd "$OGBENCH_ROOT/impls"

CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" eval_lewm_jax_cem.py \
  --task="$task" --checkpoint="$run_dir/weights_epoch_10.msgpack" \
  --data-root="$LEWM_DATA_ROOT" \
  --num-eval=50 --seed=42 --goal-offset-steps=25 --eval-budget=50 \
  --cem-horizon=5 --cem-receding-horizon=5 --action-block=5 \
  --cem-num-samples=300 --cem-steps=30 --cem-topk=30 --cem-var-scale=1.0 \
  --video-dir="$output_dir/videos" --output="$output_dir/$task.json" \
  2>&1 | tee "$output_dir/eval.log"
