#!/usr/bin/env bash
set -euo pipefail

# Server 23：评测 LeWM + GCIQL-Chunk-AWR shared-all 四任务。
# 每个任务先评测 direct pi，再评测 CEM + pi；未完成任务等待 100K checkpoint。
CLIENT_ID=23
source /home/dzb/ogbench/scripts/client_env.sh
cd "$OGBENCH_ROOT/impls"

tasks=(cube pusht reacher tworoom)
gpus=(2 3 4 5)
lewm_exps=(
  LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
  LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95
)
checkpoint_root=/data/dzb/stablewm-data/lewm-gciql-chunk-shared-runs
direct_root=/data/dzb/stablewm-data/lewm-gciql-chunk-shared-evals/20260823_rerun_direct_pi_shared_all_s100k_ep50_seed42
cem_root=/data/dzb/stablewm-data/lewm-gciql-chunk-shared-evals/20260823_cem_with_shared_all_pi_j5_h5_rh1_ep50_seed42

pids=()
for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  checkpoint_dir="$checkpoint_root/2026-08-22_23_LeWM_with_GCIQL_Chunk_AWR_shared_all_${task}_lewmseed3072_k5_bs256_s100k_s0"
  checkpoint="$checkpoint_dir/params_100000.pkl"
  direct_dir="$direct_root/$task"
  cem_dir="$cem_root/$task"
  mkdir -p "$direct_dir" "$cem_dir"

  (
    until [[ -f "$checkpoint" ]]; do
      echo "$(date '+%F %T') waiting for $checkpoint"
      sleep 60
    done
    CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" eval_ogbench_agent_lewm_envs.py \
      --task="$task" --method=gciql_chunk_lewm \
      --checkpoint-dir="$checkpoint_dir" --checkpoint-step=100000 \
      --data-root="$LEWM_DATA_ROOT" \
      --num-eval=50 --seed=42 --goal-offset-steps=25 --eval-budget=50 \
      --output="$direct_dir/$task.json" \
      >"$direct_dir/eval.log" 2>&1

    CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
    PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
    "$PYTHON_BIN" eval_lewm_jax_cem.py \
      --task="$task" \
      --checkpoint="/data/dzb/stablewm-data/lewm-jax-runs/${lewm_exps[$i]}/weights_epoch_10.msgpack" \
      --data-root="$LEWM_DATA_ROOT" \
      --num-eval=50 --seed=42 --goal-offset-steps=25 --eval-budget=50 \
      --cem-horizon=5 --cem-receding-horizon=1 --action-block=5 \
      --cem-num-samples=300 --cem-steps=5 --cem-topk=30 --cem-var-scale=1.0 \
      --cem-cost-mode=min_over_horizon --paired-plan-keys \
      --proposal-method=gciql_chunk_lewm \
      --proposal-checkpoint-dir="$checkpoint_dir" --proposal-checkpoint-step=100000 \
      --proposal-temperature=0.0 --proposal-num-samples=1 --proposal-selection=mode \
      --output="$cem_dir/$task.json" \
      >"$cem_dir/eval.log" 2>&1
  ) &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
