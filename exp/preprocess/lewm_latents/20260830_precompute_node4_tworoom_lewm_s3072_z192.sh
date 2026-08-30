#!/usr/bin/env bash
set -euo pipefail

# A800 node4：在 GPU0 上把 TwoRoom Lance 全量转换为 frozen LeWM seed3072、epoch10 的 192 维 float32 latent HDF5；支持断点续跑。
CLIENT_ID=node4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
BATCH_SIZE=${BATCH_SIZE:-512}
DECODE_WORKERS=${DECODE_WORKERS:-16}
GPU_ID=${GPU_ID:-0}
SMOKE_ROWS=${SMOKE_ROWS:-0}
source "$OGBENCH_ROOT/scripts/client_env.sh"

LEWM_DATA_ROOT=${LEWM_DATA_ROOT:-/data-training/yyf/datasets/latent-geometry}
LEWM_SEED3072_ROOT=${LEWM_SEED3072_ROOT:-/data-training/yyf/models/lewm-jax-seed3072}
LEWM_LATENT_ROOT=${LEWM_LATENT_ROOT:-/data-training/yyf/datasets/lewm-latents}
lance_path="$LEWM_DATA_ROOT/tworoom.lance"
checkpoint="$LEWM_SEED3072_ROOT/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"
output="$LEWM_LATENT_ROOT/tworoom__lewm_s3072_e10_z192.h5"
log="$LEWM_LATENT_ROOT/tworoom__lewm_s3072_e10_z192.log"

mkdir -p "$LEWM_LATENT_ROOT"
cd "$OGBENCH_ROOT/impls"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_PLATFORMS=cuda \
PYTHONUNBUFFERED=1 \
PYTHONPATH="$OGBENCH_ROOT:$OGBENCH_ROOT/impls" \
"$PYTHON_BIN" precompute_lewm_latents.py \
  --task=tworoom \
  --lance-path="$lance_path" \
  --checkpoint="$checkpoint" \
  --output="$output" \
  --batch-size="$BATCH_SIZE" \
  --decode-workers="$DECODE_WORKERS" \
  --output-dtype=float32 \
  --flush-every-batches=20 \
  --log-every-batches=20 \
  --smoke-rows="$SMOKE_ROWS" \
  2>&1 | tee -a "$log"
