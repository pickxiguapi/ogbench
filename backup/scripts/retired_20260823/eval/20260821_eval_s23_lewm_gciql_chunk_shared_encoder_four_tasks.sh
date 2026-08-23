#!/usr/bin/env bash
set -euo pipefail

# Server 23：四卡并行评测 shared-encoder GCIQL-Chunk + LeWM；actor 仍用独立 AWR checkpoint，共享 LeWM latent 的 Q 每轮保留 150/300 候选，再由 min-over-horizon LeWM cost 选择 30 个 elite；J5、H5、RH1、action-block5、50 episodes、seed42。
CLIENT_ID=23
source /home/dzb/ogbench-shared-q/scripts/client_env.sh
OGBENCH_ROOT=/home/dzb/ogbench-shared-q
PYTHON_BIN=/home/dzb/ogbench/.venv/bin/python
cd "$OGBENCH_ROOT/impls"

tasks=(cube pusht reacher tworoom)
lewm_exps=(
  LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
)
gpus=(0 3 4 5)
shared_root=/data/dzb/stablewm-data/gciql-chunk-lewm-shared-runs
output_root=/data/dzb/stablewm-data/lewm-jax-shared-q-evals/20260821_shared_q_keep150_mincost_j5_h5_rh1_ab5_seed42

pids=()
for i in "${!tasks[@]}"; do
  shared_dir="$shared_root/20260821_s23_${tasks[$i]}_lewm_shared_qv_gciql_chunk_awr_actor_k5_bs256_s100k_s0_e09"
  output_dir="$output_root/${tasks[$i]}"
  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" eval_lewm_jax_cem.py \
    --task="${tasks[$i]}" \
    --checkpoint="/data/dzb/stablewm-data/lewm-jax-runs/${lewm_exps[$i]}/weights_epoch_10.msgpack" \
    --data-root="$LEWM_DATA_ROOT" \
    --num-eval=50 --seed=42 --goal-offset-steps=25 --eval-budget=50 \
    --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
    --cem-num-samples=300 --cem-steps=5 --cem-topk=30 --cem-var-scale=1.0 \
    --cem-cost-mode=min_over_horizon --paired-plan-keys \
    --proposal-method=gciql_chunk \
    --proposal-checkpoint-dir="/data/dzb/stablewm-data/gciql-chunk-proposals-s11/${tasks[$i]}" \
    --proposal-checkpoint-step=100000 --proposal-temperature=0.0 \
    --native-q-keep=150 \
    --shared-q-checkpoint-dir="$shared_dir" --shared-q-checkpoint-step=100000 \
    --output="$output_dir/${tasks[$i]}.json" \
    >"$output_dir/eval.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
