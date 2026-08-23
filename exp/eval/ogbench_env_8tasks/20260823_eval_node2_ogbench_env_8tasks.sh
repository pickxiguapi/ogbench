#!/usr/bin/env bash
set -euo pipefail

# A800 node2：八卡并行评测 OGBench-Env-8Tasks；MODE 可设 policy/lewm/guided/native_q，REPRESENTATION_MODE 可设 independent/pi/qv/all。
CLIENT_ID=node2
MODE=${MODE:-guided}
REPRESENTATION_MODE=${REPRESENTATION_MODE:-independent}
POLICY_STEPS=${POLICY_STEPS:-500000}
POLICY_SEED=${POLICY_SEED:-0}
POLICY_BATCH_SIZE=${POLICY_BATCH_SIZE:-512}
P_AUG=${P_AUG:-0.0}
LEWM_STEPS=${LEWM_STEPS:-200000}
LEWM_SEED=${LEWM_SEED:-3072}
LEWM_BATCH_SIZE=${LEWM_BATCH_SIZE:-128}
NUM_EVAL=${NUM_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-42}
CEM_HORIZON=${CEM_HORIZON:-5}
CEM_NUM_SAMPLES=${CEM_NUM_SAMPLES:-300}
CEM_STEPS=${CEM_STEPS:-5}
CEM_TOPK=${CEM_TOPK:-30}
PROPOSAL_NUM_SAMPLES=${PROPOSAL_NUM_SAMPLES:-64}
PROPOSAL_TEMPERATURE=${PROPOSAL_TEMPERATURE:-0.1}
EVAL_TAG=${EVAL_TAG:-p${POLICY_STEPS}_w${LEWM_STEPS}_cem${CEM_NUM_SAMPLES}x${CEM_STEPS}_h${CEM_HORIZON}}
source /home/yyf/ogbench-main/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

case "$MODE" in policy|lewm|guided|native_q) ;; *) echo "MODE must be policy, lewm, guided, or native_q" >&2; exit 2 ;; esac
case "$REPRESENTATION_MODE" in independent|pi|qv|all) ;; *) echo "REPRESENTATION_MODE must be independent, pi, qv, or all" >&2; exit 2 ;; esac
envs=(visual-cube-single-play-v0 visual-cube-double-play-v0 visual-cube-triple-play-v0 visual-scene-play-v0 visual-cube-single-noisy-v0 visual-cube-double-noisy-v0 visual-cube-triple-noisy-v0 visual-scene-noisy-v0)
tags=(cs_play cd_play ct_play scene_play cs_noisy cd_noisy ct_noisy scene_noisy)
output_root="$CLIENT_ROOT/lewm-final/evals/ogbench-env-8tasks/${MODE}_${REPRESENTATION_MODE}_${EVAL_TAG}_seed${EVAL_SEED}"
pids=()

for i in "${!envs[@]}"; do
  lewm_dir="$CLIENT_ROOT/lewm-final/lewm-ogbench8/lewm_ogbench8_${tags[$i]}_s${LEWM_STEPS}_bs${LEWM_BATCH_SIZE}_s${LEWM_SEED}"
  policy_dir="$CLIENT_ROOT/lewm-final/gciql-chunk-ogbench8/gciql_chunk_ogbench8_${tags[$i]}_${REPRESENTATION_MODE}_s${POLICY_STEPS}_bs${POLICY_BATCH_SIZE}_paug${P_AUG}_s${POLICY_SEED}"
  args=()
  [[ "$MODE" != policy ]] && args+=(--lewm-checkpoint="$lewm_dir/weights_step_${LEWM_STEPS}.msgpack")
  [[ "$MODE" != lewm ]] && args+=(--policy-checkpoint-dir="$policy_dir" --policy-checkpoint-step="$POLICY_STEPS")
  output_dir="$output_root/${tags[$i]}"
  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES=$i XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" eval_ogbench_env_8tasks.py \
    --env-name="${envs[$i]}" --dataset-path="$OGBENCH_DATA_DIR/${envs[$i]}.npz" \
    --mode="$MODE" "${args[@]}" --policy-action-space=environment \
    --num-eval="$NUM_EVAL" --seed="$EVAL_SEED" \
    --cem-horizon="$CEM_HORIZON" --cem-receding-horizon=1 --action-block=5 \
    --cem-num-samples="$CEM_NUM_SAMPLES" --cem-steps="$CEM_STEPS" --cem-topk="$CEM_TOPK" --cem-var-scale=1.0 \
    --proposal-num-samples="$PROPOSAL_NUM_SAMPLES" --proposal-temperature="$PROPOSAL_TEMPERATURE" \
    --output="$output_dir/result.json" >"$output_dir/eval.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
