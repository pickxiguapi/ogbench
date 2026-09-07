#!/usr/bin/env bash

LEWMPP_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OGBENCH_ROOT=$(cd "$LEWMPP_SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

require_file() {
  local path=$1
  local label=$2
  if [[ ! -s "$path" ]]; then
    echo "Missing $label: $path" >&2
    return 2
  fi
}

verify_generator() {
  local family=$1
  local generator_type=$2
  local checkpoint=$3
  "$PYTHON_BIN" - "$family" "$generator_type" "$checkpoint" <<'PY'
import json
import pathlib
import sys

family, generator_type, checkpoint_arg = sys.argv[1:]
checkpoint = pathlib.Path(checkpoint_arg).expanduser().resolve()
config_path = checkpoint.parent / 'config.json'
if not checkpoint.is_file():
    raise SystemExit(f'Missing subgoal-generator checkpoint: {checkpoint}')
if not config_path.is_file():
    raise SystemExit(f'Missing adjacent generator config: {config_path}')
config = json.loads(config_path.read_text())
expected = {
    'goalmax25': (
        'uniform_distance_first_aligned_future_same_trajectory_stride_5_max_25',
        25,
    ),
    'general_uniform_future': ('hiql_uniform_future_same_trajectory', None),
}
architectures = {
    'mlp': ('history_latent_mlp', 'direct_latent_mlp_512x3'),
    'endpoint_flow': ('latent_endpoint_flow_transformer_encoder',),
    'latent_path_flow': ('latent_path_flow_transformer_encoder',),
}
if family not in expected:
    raise SystemExit(f'Unsupported generator family: {family}')
sampling, max_steps = expected[family]
if config.get('goal_sampling') != sampling:
    raise SystemExit(
        f'{family} goal_sampling mismatch: {config.get("goal_sampling")!r} != {sampling!r}'
    )
if config.get('max_goal_steps') != max_steps:
    raise SystemExit(
        f'{family} max_goal_steps mismatch: {config.get("max_goal_steps")!r} != {max_steps!r}'
    )
if generator_type not in architectures:
    raise SystemExit(f'Unsupported generator type: {generator_type}')
if config.get('architecture') not in architectures[generator_type]:
    raise SystemExit(
        f'{generator_type} architecture mismatch: '
        f'{config.get("architecture")!r} not in {architectures[generator_type]!r}'
    )
print(f'verified {family}/{generator_type}: {checkpoint}')
PY
}
