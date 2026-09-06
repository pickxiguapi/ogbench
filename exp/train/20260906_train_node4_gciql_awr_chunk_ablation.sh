#!/usr/bin/env bash
set -euo pipefail

# A800 node4: train the four LeWM benchmark policies for one GCIQL-Chunk-AWR
# action chunk size.  Run this script once for c=1 and once for c=10 on
# disjoint four-GPU groups.  Relative to the canonical c=5 policy, only
# --chunk_size and the output name change.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

CHUNK_SIZE=${CHUNK_SIZE:?Set CHUNK_SIZE to 1 or 10}
GPU_IDS=${GPU_IDS:?Set GPU_IDS to four whitespace-separated GPU IDs}
POLICY_SEED=${POLICY_SEED:-777}
POLICY_STEPS=${POLICY_STEPS:-100000}
POLICY_BATCH_SIZE=${POLICY_BATCH_SIZE:-256}
P_AUG=${P_AUG:-0.0}
PYTHON_BIN=${PYTHON_BIN:-/data-training/yyf/envs/ogbench/bin/python}
DATA_ROOT=${DATA_ROOT:-/data-training/yyf/datasets/latent-geometry}
RUNS_ROOT=${RUNS_ROOT:-/data-training/yyf/ogbench-lewm-policy-runs/gciql-chunk-4tasks-chunk-ablation}

if [[ "$CHUNK_SIZE" != 1 && "$CHUNK_SIZE" != 10 ]]; then
  echo "CHUNK_SIZE must be 1 or 10, got: $CHUNK_SIZE" >&2
  exit 2
fi

read -r -a gpus <<< "$GPU_IDS"
if (( ${#gpus[@]} != 4 )); then
  echo "GPU_IDS must contain exactly four whitespace-separated GPU IDs." >&2
  exit 2
fi

datasets=(cube_single_expert pusht_expert_train reacher tworoom)
tags=(cube pusht reacher tworoom)

lewm_seed3072_root=/data-training/yyf/models/lewm-jax-seed3072
lewm_seed666_root=/data-training/yyf/models/lewm-jax-seed666
lewm_checkpoints=(
  "$lewm_seed3072_root/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"
  "$lewm_seed666_root/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack"
  "$lewm_seed3072_root/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"
  "$lewm_seed3072_root/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"
)

[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
for i in "${!datasets[@]}"; do
  dataset="$DATA_ROOT/${datasets[$i]}.lance"
  checkpoint=${lewm_checkpoints[$i]}
  [[ -e "$dataset" ]] || { echo "Dataset not found: $dataset" >&2; exit 2; }
  [[ -f "$checkpoint" ]] || { echo "Frozen LeWM checkpoint not found: $checkpoint" >&2; exit 2; }

  exp_name="gc4_${tags[$i]}_all_awr_c${CHUNK_SIZE}_n${POLICY_STEPS}_b${POLICY_BATCH_SIZE}_a${P_AUG}_sd${POLICY_SEED}"
  run_dir="$RUNS_ROOT/$exp_name"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite existing run directory: $run_dir" >&2
    exit 2
  fi
done

for gpu in "${gpus[@]}"; do
  if [[ -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
    echo "GPU $gpu is busy; refusing to launch." >&2
    exit 2
  fi
done

mkdir -p "$RUNS_ROOT"
cd "$OGBENCH_ROOT/impls"

echo "[$(date '+%F %T %Z')] starting GCIQL-Chunk-AWR c=$CHUNK_SIZE seed=$POLICY_SEED on GPUs ${gpus[*]}"
pids=()
for i in "${!datasets[@]}"; do
  exp_name="gc4_${tags[$i]}_all_awr_c${CHUNK_SIZE}_n${POLICY_STEPS}_b${POLICY_BATCH_SIZE}_a${P_AUG}_sd${POLICY_SEED}"
  run_dir="$RUNS_ROOT/$exp_name"
  mkdir -p "$run_dir"

  CUDA_VISIBLE_DEVICES=${gpus[$i]} XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
  "$PYTHON_BIN" train_gciql_chunk.py \
    --dataset_path="$DATA_ROOT/${datasets[$i]}.lance" \
    --save_dir="$run_dir" --representation_mode=all \
    --lewm_checkpoint="${lewm_checkpoints[$i]}" \
    --actor_loss=awr --alpha=3.0 --pixel_encoder=impala_small --p_aug="$P_AUG" \
    --train_steps="$POLICY_STEPS" --batch_size="$POLICY_BATCH_SIZE" --seed="$POLICY_SEED" \
    --chunk_size="$CHUNK_SIZE" --lr=3e-4 --discount=0.99 --expectile=0.9 --tau=0.005 \
    --log_interval=5000 --save_interval="$POLICY_STEPS" >"$run_dir/train.log" 2>&1 &
  pids+=("$!")
  echo "task=${tags[$i]} gpu=${gpus[$i]} pid=${pids[-1]} output=$run_dir"
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done

echo "[$(date '+%F %T %Z')] c=$CHUNK_SIZE finished with status=$status"
exit "$status"
