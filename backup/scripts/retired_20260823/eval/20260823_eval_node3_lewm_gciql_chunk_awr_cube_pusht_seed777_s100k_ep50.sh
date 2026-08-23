#!/usr/bin/env bash
set -euo pipefail

# node3：等待 Cube/PushT 的普通 GCIQL-Chunk-AWR seed777 训练到 100k，分别用 GPU 6/7 评测 50 episodes（eval seed42、goal offset25、budget50）。
CLIENT_ID=node3
source scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

tasks=(cube pusht)
gpus=(6 7)
checkpoints=(
  "$CLIENT_ROOT/ogbench-lewm-policy-runs/2026-08-22_node3_GCAWR_lewm_cube_k5_bs256_s100k_s777_a3_e09_aug05/OGBench/GCAWR_lewm_cube_s777/sd777_20260822_122737"
  "$CLIENT_ROOT/ogbench-lewm-policy-runs/2026-08-22_node3_GCAWR_lewm_pusht_k5_bs256_s100k_s777_a3_e09_aug05/OGBench/GCAWR_lewm_pusht_s777/sd777_20260822_122737"
)
output_root="$CLIENT_ROOT/ogbench-lewm-policy-runs/evals/2026-08-23_node3_GCAWR_seed777_s100k_ep50_seed42_go25_b50"

for i in "${!tasks[@]}"; do
  (
    until [[ -s "${checkpoints[$i]}/params_100000.pkl" ]]; do sleep 60; done
    mkdir -p "$output_root/${tasks[$i]}"
    CUDA_VISIBLE_DEVICES="${gpus[$i]}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    EGL_PLATFORM=surfaceless \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" eval_ogbench_agent_lewm_envs.py \
      --task "${tasks[$i]}" \
      --method gciql_chunk \
      --checkpoint-dir "${checkpoints[$i]}" \
      --checkpoint-step 100000 \
      --data-root "$CLIENT_ROOT/datasets/lewm" \
      --num-eval 50 \
      --seed 42 \
      --goal-offset-steps 25 \
      --eval-budget 50 \
      --output "$output_root/${tasks[$i]}/results.json" \
      > "$output_root/${tasks[$i]}/eval.log" 2>&1
  ) &
done
wait
