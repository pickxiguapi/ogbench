#!/usr/bin/env bash
set -euo pipefail

# Fill in these paths before running this script.
OGBENCH_DATA_ROOT=""
EXPERIMENT_ROOT="outputs"
LEWM_CHECKPOINT_ROOT=""
ACTION_PRIOR_CHECKPOINT_ROOT=""
LATENT_PATH_FLOW_CHECKPOINT_ROOT=""
EVAL_ROOT="$EXPERIMENT_ROOT/evals/visual-ogbench8/lewmpp"
ENVS=(
  visual-cube-single-play-v0
  visual-cube-double-play-v0
  visual-cube-triple-play-v0
  visual-scene-play-v0
  visual-cube-single-noisy-v0
  visual-cube-double-noisy-v0
  visual-cube-triple-noisy-v0
  visual-scene-noisy-v0
)
TAGS=(cs_play cd_play ct_play scene_play cs_noisy cd_noisy ct_noisy scene_noisy)
GPU_IDS=(0 1 2 3 4 5 6 7)
EVAL_SEEDS=(0 1 42)
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless

for index in "${!ENVS[@]}"; do
  tag=${TAGS[$index]}
  lewm="$LEWM_CHECKPOINT_ROOT/$tag/weights_epoch_10.msgpack"
  generator="$LATENT_PATH_FLOW_CHECKPOINT_ROOT/$tag/checkpoint_200000.msgpack"
  python experiments/eval/validate_visual_ogbench8_checkpoint.py \
    --generator="$generator" --lewm="$lewm"
done

eval_one() {
  local gpu=$1 index=$2 env_name=${ENVS[$2]} tag=${TAGS[$2]}
  local output_dir="$EVAL_ROOT/seed$CURRENT_EVAL_SEED/$tag"
  local output="$output_dir/result.json"
  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python impls/eval_ogbench_env_8tasks.py \
      --env-name="$env_name" --dataset-path="$OGBENCH_DATA_ROOT/$env_name.npz" \
      --controller=lewm_cem --policy-guidance=policy_random_mixture \
      --guidance-population-size=250 --guidance-temperature=0.05 \
      --guidance-first-block-std=0.05 --guidance-random-elite-cap=5 \
      --guidance-mean-residual-weight=0.5 --guidance-goal-mode=final \
      --use-subgoal --latent-subgoal-checkpoint="$LATENT_PATH_FLOW_CHECKPOINT_ROOT/$tag/checkpoint_200000.msgpack" \
      --num-samples=1 --lewm-checkpoint="$LEWM_CHECKPOINT_ROOT/$tag/weights_epoch_10.msgpack" \
      --policy-checkpoint-dir="$ACTION_PRIOR_CHECKPOINT_ROOT/$tag" --policy-checkpoint-step=500000 \
      --policy-action-space=environment --num-eval=50 --seed="$CURRENT_EVAL_SEED" \
      --cem-horizon=2 --cem-receding-horizon=1 --action-block=5 \
      --cem-num-samples=300 --cem-iterations=5 --cem-topk=30 \
      --cem-var-scale=1.0 --cem-cost-mode=moh --output="$output" \
      2>&1 | tee "$output_dir/eval.log"
}

for CURRENT_EVAL_SEED in "${EVAL_SEEDS[@]}"; do
  export CURRENT_EVAL_SEED
  pids=()
  for index in "${!ENVS[@]}"; do
    eval_one "${GPU_IDS[$index]}" "$index" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  (( status == 0 )) || exit "$status"
done

python impls/aggregate_visual_ogbench8_results.py \
  --results-root="$EVAL_ROOT" --output="$EVAL_ROOT/summary.json"
