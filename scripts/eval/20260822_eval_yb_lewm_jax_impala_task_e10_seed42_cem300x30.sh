#!/usr/bin/env bash
set -euo pipefail

# 英博云：评测训练 seed42 的指定 LeWM-JAX IMPALA epoch10；50 episodes、eval seed42、dataset-goal、CEM 300×30。
task="${1:-}"
CLIENT_ID=yb
source /root/data/yyf/ogbench-new/scripts/client_env.sh

case "$task" in
  cube) gpu=4 ;;
  pusht) gpu=5 ;;
  reacher) gpu=6 ;;
  tworoom) gpu=7 ;;
  *)
    echo "Usage: bash scripts/eval/20260822_eval_yb_lewm_jax_impala_task_e10_seed42_cem300x30.sh {cube|pusht|reacher|tworoom}" >&2
    exit 2
    ;;
esac

run_dir="$CLIENT_ROOT/lewm-jax-runs/2026-08-21_yb_LeWMJAX_${task}_impalasmall_lance_bs128_e10_s42_fs5_h3_sigreg009"
output_dir="$run_dir/eval_cem300x30_seed42_20260822"
mkdir -p "$output_dir/videos"
cd "$OGBENCH_ROOT/impls"

CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" eval_lewm_jax_cem.py \
  --task="$task" --checkpoint="$run_dir/weights_epoch_10.msgpack" \
  --data-root="$LEWM_DATA_ROOT" \
  --num-eval=50 --seed=42 --goal-offset-steps=25 --eval-budget=50 \
  --cem-horizon=5 --cem-receding-horizon=5 --action-block=5 \
  --cem-num-samples=300 --cem-steps=30 --cem-topk=30 --cem-var-scale=1.0 \
  --video-dir="$output_dir/videos" --output="$output_dir/$task.json" \
  2>&1 | tee "$output_dir/eval.log"
