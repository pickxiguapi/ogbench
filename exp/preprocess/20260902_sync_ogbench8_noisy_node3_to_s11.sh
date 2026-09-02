#!/usr/bin/env bash
set -euo pipefail

# Run locally: copy the four OGBench noisy train/validation NPZ pairs from
# A800 node3 to server11. Each file uses four disjoint SSH streams, is checked
# with SHA256, and only becomes visible under its final name after validation.
SOURCE_HOST=${SOURCE_HOST:-a800-node3}
DEST_HOST=${DEST_HOST:-yyf@172.28.102.11}
SOURCE_ROOT=${SOURCE_ROOT:-/data-training/yyf/ogbench-cache/data}
DEST_ROOT=${DEST_ROOT:-/data/yyf/H-LeWM/ogbench-cache/data}
STREAMS=${STREAMS:-4}
BLOCK_SIZE=${BLOCK_SIZE:-1048576}

files=(
  visual-cube-single-noisy-v0.npz
  visual-cube-single-noisy-v0-val.npz
  visual-cube-double-noisy-v0.npz
  visual-cube-double-noisy-v0-val.npz
  visual-cube-triple-noisy-v0.npz
  visual-cube-triple-noisy-v0-val.npz
  visual-scene-noisy-v0.npz
  visual-scene-noisy-v0-val.npz
)

if (( STREAMS < 1 )); then
  echo "STREAMS must be positive." >&2
  exit 2
fi

copy_file() {
  local name=$1
  local source_path="$SOURCE_ROOT/$name"
  local dest_path="$DEST_ROOT/$name"
  local temp_path="$dest_path.tmp"
  local source_size dest_size source_sha dest_sha total_blocks chunk_blocks

  source_size=$(ssh "$SOURCE_HOST" "stat -c %s '$source_path'")
  if [[ ! "$source_size" =~ ^[0-9]+$ ]] || (( source_size < 1 )); then
    echo "Invalid or missing source file: $source_path" >&2
    exit 2
  fi

  dest_size=$(ssh "$DEST_HOST" "if [[ -f '$dest_path' ]]; then stat -c %s '$dest_path'; fi")
  if [[ -n "$dest_size" ]]; then
    if [[ "$dest_size" != "$source_size" ]]; then
      echo "Existing destination has wrong size; refusing to overwrite: $dest_path" >&2
      exit 2
    fi
    source_sha=$(ssh "$SOURCE_HOST" "sha256sum '$source_path'" | awk '{print $1}')
    dest_sha=$(ssh "$DEST_HOST" "sha256sum '$dest_path'" | awk '{print $1}')
    if [[ "$source_sha" != "$dest_sha" ]]; then
      echo "Existing destination checksum mismatch: $dest_path" >&2
      exit 2
    fi
    echo "[$(date '+%F %T %Z')] Already complete: $name"
    return
  fi

  total_blocks=$(( (source_size + BLOCK_SIZE - 1) / BLOCK_SIZE ))
  chunk_blocks=$(( (total_blocks + STREAMS - 1) / STREAMS ))
  ssh "$DEST_HOST" "mkdir -p '$DEST_ROOT' && truncate -s 0 '$temp_path'"
  echo "[$(date '+%F %T %Z')] Copy $name ($source_size bytes, $STREAMS streams)"

  local pids=()
  local stream offset
  for (( stream = 0; stream < STREAMS; stream++ )); do
    offset=$(( stream * chunk_blocks ))
    if (( offset >= total_blocks )); then
      break
    fi
    (
      ssh "$SOURCE_HOST" \
        "dd if='$source_path' bs=$BLOCK_SIZE skip=$offset count=$chunk_blocks iflag=fullblock status=none" \
      | ssh "$DEST_HOST" \
        "dd of='$temp_path' bs=$BLOCK_SIZE seek=$offset conv=notrunc status=none"
    ) &
    pids+=("$!")
  done

  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if (( status != 0 )); then
    echo "Segmented transfer failed: $name" >&2
    exit 1
  fi

  source_sha=$(ssh "$SOURCE_HOST" "sha256sum '$source_path'" | awk '{print $1}')
  dest_sha=$(ssh "$DEST_HOST" "sha256sum '$temp_path'" | awk '{print $1}')
  if [[ "$source_sha" != "$dest_sha" ]]; then
    echo "Checksum mismatch after transfer: $name" >&2
    exit 1
  fi
  ssh "$DEST_HOST" "mv '$temp_path' '$dest_path'"
  echo "[$(date '+%F %T %Z')] Verified: $name sha256=$source_sha"
}

for file in "${files[@]}"; do
  copy_file "$file"
done

echo "[$(date '+%F %T %Z')] All four noisy dataset pairs are ready on server11."
