#!/usr/bin/env bash
set -euo pipefail

# node4：评测从 node3 同步的 Cube/PushT 普通 GCIQL-Chunk-AWR seed777 100k checkpoint，GPU 0/1 各50 episodes（eval seed42、goal offset25、budget50）。
CLIENT_ID=node4
source scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

tasks=(cube pusht)
gpus=(0 1)
checkpoints=(
  "$CLIENT_ROOT/ogbench-lewm-policy-runs/node3-checkpoints/GCAWR_lewm_cube_s777_100k"
  "$CLIENT_ROOT/ogbench-lewm-policy-runs/node3-checkpoints/GCAWR_lewm_pusht_s777_100k"
)
output_root="$CLIENT_ROOT/ogbench-lewm-policy-runs/evals/2026-08-23_node3_GCAWR_seed777_s100k_ep50_seed42_go25_b50"

for i in "${!tasks[@]}"; do
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
    --data-root "$LEWM_DATA_ROOT" \
    --num-eval 50 \
    --seed 42 \
    --goal-offset-steps 25 \
    --eval-budget 50 \
    --output "$output_root/${tasks[$i]}/results.json" \
    > "$output_root/${tasks[$i]}/eval.log" 2>&1 &
done
wait
