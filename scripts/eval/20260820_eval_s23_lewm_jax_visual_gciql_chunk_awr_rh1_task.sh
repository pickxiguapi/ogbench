#!/usr/bin/env bash
set -euo pipefail

# Server 23: GCIQL-Chunk-AWR proposes the first block for LeWM CEM.
# Plan five action blocks, execute one block, then replan from the new image.
task="${1:-}"
CLIENT_ID=23
source /home/dzb/ogbench/scripts/client_env.sh

case "$task" in
  single)
    env_name=visual-cube-single-play-v0
    tag=visual_cube_single
    gpu=2
    ;;
  double)
    env_name=visual-cube-double-play-v0
    tag=visual_cube_double
    gpu=3
    ;;
  triple)
    env_name=visual-cube-triple-play-v0
    tag=visual_cube_triple
    gpu=4
    ;;
  scene)
    env_name=visual-scene-play-v0
    tag=visual_scene
    gpu=5
    ;;
  *)
    echo "Usage: bash scripts/eval/20260820_eval_s23_lewm_jax_visual_gciql_chunk_awr_rh1_task.sh {single|double|triple|scene}" >&2
    exit 2
    ;;
esac

lewm_exp="LeWMJAX_ogbench_${tag}_play_impalasmall_bs512_s500k_seed3072_fs5_chunk5_cemh5_bf16_s23"
run_dir="/data/dzb/stablewm-data/lewm-jax-runs/$lewm_exp"
proposal_dir="/data/dzb/stablewm-data/gciql-chunk-awr-visual/$task"
output_dir="$run_dir/eval_gciql_chunk_awr_s500k_cem300x30_h5_rh1_ab5_seed42"

test -s "$run_dir/weights_step_500000.msgpack"
test -s "$proposal_dir/params_500000.pkl"
test -s "$proposal_dir/flags.json"
mkdir -p "$output_dir"
cd "$OGBENCH_ROOT/impls"

CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" eval_lewm_jax_ogbench.py \
  --env-name="$env_name" --dataset-path="/home/dzb/.ogbench/data/$env_name.npz" \
  --checkpoint="$run_dir/weights_step_500000.msgpack" \
  --num-eval=50 --seed=42 \
  --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
  --cem-num-samples=300 --cem-steps=30 --cem-topk=30 \
  --proposal-method=gciql_chunk \
  --proposal-checkpoint-dir="$proposal_dir" --proposal-checkpoint-step=500000 \
  --proposal-temperature=0.0 \
  --output="$output_dir/$task.json" 2>&1 | tee "$output_dir/eval.log"
