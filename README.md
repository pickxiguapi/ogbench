# LeWM++

LeWM++ is a closed-loop latent-space planner for long-distance image-goal control. It combines a frozen LeWM world model with three independently switchable components:

1. a subgoal generator that maps observation history and the remote final goal to a reachable latent target;
2. a final-goal-conditioned GCIQL-AWR-Chunk action prior that initializes CEM;
3. min-over-horizon (MoH) trajectory scoring.

This release branch reproduces the LeWM four-task experiments at goal offsets H25, H50, H75, and H100. It also provides the three paper ablations, standalone LeWM and GCIQL-AWR-Chunk baselines, three subgoal-generator architectures, and two generator sampling families.

The code is based on [OGBench](https://github.com/seohongpark/ogbench) and retains its MIT license. The preliminary DINO-WM transfer study uses a separate codebase and is not included here.

## Supported experiment matrix

The unified evaluator accepts these variants:

| `VARIANT` | Subgoal | Action prior | Cost | Controller |
|---|---|---|---|---|
| `full` | on | selectable policy mode | MoH | LeWM++ |
| `no_subgoal` | off | selectable policy mode | MoH | LeWM++ w/o Subgoal Path |
| `no_action_prior` | on | zero initialization | MoH | LeWM++ w/o Action Prior |
| `no_moh` | on | selectable policy mode | terminal | LeWM++ w/o MoH |
| `lewm` | off | zero initialization | terminal | standalone LeWM CEM |
| `gciql_chunk` | off | direct execution | n/a | standalone GCIQL-AWR-Chunk |

The public subgoal interface supports:

| `GENERATOR_TYPE` | Prediction | Training objective |
|---|---|---|
| `mlp` | one endpoint | latent regression |
| `endpoint_flow` | one endpoint | conditional flow matching |
| `latent_path_flow` | chunk-aligned path | conditional path flow matching |

Every type supports both `goalmax25` and `general_uniform_future` training. The paper's main LeWM++ results use `latent_path_flow`.

Action-prior initialization is selected with `ACTION_PRIOR_MODE`:

- `zero`: initialize the complete CEM plan at zero and do not load a policy;
- `policy_mode`: initialize the first action block with the deterministic policy mode;
- `policy_mode_anchor`: use the same initialization and preserve that original policy plan as a CEM candidate in every iteration.

In all modes, the policy input is always the original final goal. A generated subgoal is used only by the LeWM rollout cost and is never passed to the policy.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/pickxiguapi/ogbench.git
cd ogbench
git switch release/lewmpp-open-source
uv sync --extra train --extra dev
```

On Linux with CUDA 12:

```bash
uv sync --extra train --extra cuda12 --extra dev
```

All training and evaluation jobs are launched through the Bash files in `exp/lewmpp/`. Activate `.venv`, or invoke a launcher as `uv run bash exp/lewmpp/<script>.sh`.

## Data and path configuration

The four tasks are Cube, PushT, Reacher, and TwoRoom. Each task needs an evaluation HDF5 file, a JPEG-backed Lance table, a frozen LeWM checkpoint, a GCIQL-AWR-Chunk checkpoint, and the relevant generator checkpoints.

```bash
cp configs/lewmpp_paths.example.env configs/lewmpp_paths.env
source configs/lewmpp_paths.env
```

`configs/lewmpp_paths.env` is ignored by Git. Checkpoints are not committed.

The horizon-to-family mapping is strict:

| Evaluation horizon | Family | Required `goal_sampling` | Required `max_goal_steps` |
|---|---|---|---:|
| H25 | `goalmax25` | `uniform_distance_first_aligned_future_same_trajectory_stride_5_max_25` | `25` |
| H50/H75/H100 | `general_uniform_future` | `hiql_uniform_future_same_trajectory` | unset / `null` |

Before evaluation, the Bash launcher reads the `config.json` adjacent to every generator checkpoint and aborts on a family or architecture mismatch. Output paths contain experiment group, generator family, generator type, horizon, variant, prior mode, seed, and task, so incompatible results cannot overwrite or silently pool.

## Train the components

Train the frozen LeWM:

```bash
DATASET_PATH=/data/cube.lance \
OUTPUT_DIR=/runs/lewm/cube \
TRAIN_SEED=3072 \
bash exp/lewmpp/train_lewm.sh
```

Build the checkpoint-bound latent cache:

```bash
TASK=cube \
LANCE_PATH=/data/cube.lance \
LEWM_CHECKPOINT=/runs/lewm/cube/weights_epoch_10.msgpack \
OUTPUT_PATH=/data/latents/cube.h5 \
bash exp/lewmpp/precompute_latents.sh
```

Train any generator type and family by changing two variables:

```bash
TASK=cube \
GENERATOR_TYPE=latent_path_flow \
FAMILY=goalmax25 \
LATENT_DATASET=/data/latents/cube.h5 \
OUTPUT_ROOT=/runs/subgoal-generators \
bash exp/lewmpp/train_subgoal_generator.sh
```

Valid generator types are `mlp`, `endpoint_flow`, and `latent_path_flow`; valid families are `goalmax25` and `general_uniform_future`. Flow models train and sample with 16 Euler steps by default.

Train the final-goal GCIQL-AWR-Chunk model:

```bash
TASK=cube \
DATASET_PATH=/data/cube.lance \
LEWM_CHECKPOINT=/runs/lewm/cube/weights_epoch_10.msgpack \
OUTPUT_ROOT=/runs/action-prior \
bash exp/lewmpp/train_action_prior.sh
```

The release configuration uses chunk size 5, AWR, seed 777, and a shared frozen LeWM representation for Q, V, and policy.

## Run one evaluation

The default planner uses 300 candidates, 5 CEM iterations, 30 elites, planning horizon `P=2`, receding horizon `R=1`, action chunk `c=5`, and 16 flow steps.

```bash
TASK=cube \
VARIANT=full \
EXPERIMENT_GROUP=main_h25 \
GOAL_OFFSET_STEPS=25 \
EVAL_SEED=0 \
GENERATOR_TYPE=latent_path_flow \
ACTION_PRIOR_MODE=policy_mode \
LEWM_CHECKPOINT="$LEWM_CUBE_CHECKPOINT" \
POLICY_CHECKPOINT_DIR="$POLICY_CUBE_CHECKPOINT_DIR" \
SUBGOAL_GENERATOR_CHECKPOINT="$GOALMAX25_CUBE_CHECKPOINT" \
bash exp/lewmpp/evaluate.sh
```

Useful switches are:

```bash
# Same model with the anchored policy mode.
ACTION_PRIOR_MODE=policy_mode_anchor bash exp/lewmpp/evaluate.sh

# Use another trained subgoal model.
GENERATOR_TYPE=endpoint_flow \
SUBGOAL_GENERATOR_CHECKPOINT=/runs/endpoint_flow/checkpoint_200000.msgpack \
bash exp/lewmpp/evaluate.sh

# The three ablations and two baselines.
VARIANT=no_subgoal bash exp/lewmpp/evaluate.sh
VARIANT=no_action_prior bash exp/lewmpp/evaluate.sh
VARIANT=no_moh bash exp/lewmpp/evaluate.sh
VARIANT=lewm bash exp/lewmpp/evaluate.sh
VARIANT=gciql_chunk bash exp/lewmpp/evaluate.sh
```

For variants that do not use a component, its checkpoint variable may remain set; the launcher omits it from the Python command and records the component as disabled.

## Reproduce the paper matrix

After filling and sourcing `configs/lewmpp_paths.env`:

```bash
bash exp/lewmpp/reproduce_paper.sh
```

The wrapper runs:

- full LeWM++ at H25/H50/H75/H100;
- the H25 `no_subgoal`, `no_action_prior`, and `no_moh` matched ablation blocks;
- standalone LeWM and GCIQL-AWR-Chunk baselines at all four offsets.

Set `RUN_LONG_HORIZON=0` or `RUN_BASELINES=0` to skip those groups. The H25 action-prior and subgoal blocks use evaluation seeds `{0,1,42}`; the MoH block uses `{0,1,666}`, matching the reported runs. Each full-model rerun remains in its own experiment group.

Aggregate completed JSON files without pooling groups, families, architectures, or prior modes:

```bash
uv run python impls/aggregate_lewmpp_results.py \
  --results-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/summary.csv"
```

## Reported H25 ablations

The paper records three independently rerun, matched blocks over 50 episodes per task and evaluation seed:

| Block | Full LeWM++ | Ablation | Macro change |
|---|---:|---:|---:|
| Action prior | 97.00 ± 1.08 | 92.00 ± 0.82 | +5.00 |
| Subgoal path | 97.00 ± 0.41 | 90.67 ± 0.24 | +6.33 |
| MoH score | 96.83 ± 0.62 | 94.00 ± 1.08 | +2.83 |

The deviations are population standard deviations over evaluation seeds with fixed training checkpoints; they are not training-seed uncertainty or confidence intervals.

## Verification

```bash
uv run pytest -q
uv run ruff check impls
uv run ruff format --check impls
for script in exp/lewmpp/*.sh; do bash -n "$script"; done
```

Tests cover the three generator shapes, family validation, consecutive-frame history, zero/policy/policy-anchor initialization, final-goal-only policy conditioning, MoH/terminal scoring, direct GCIQL chunk execution, and the reduced release surface.

## Repository layout

```text
configs/                    Local path template
exp/lewmpp/                 Public training/evaluation launchers
impls/action_prior.py       GCIQL-AWR-Chunk loader and direct policy
impls/subgoal_generators.py MLP, Endpoint Flow, and LatentPath Flow
impls/lewm_jax/planner.py   Canonical LeWM++ controller
impls/train_*.py            LeWM, action-prior, and generator trainers
impls/eval_lewm_4tasks.py   Unified experiment entrypoint
ogbench/                    OGBench environments and data APIs
```

## License and citation

The repository retains OGBench's MIT license. Please cite OGBench and LeWorldModel when using their benchmark, environments, or model implementation. A LeWM++ citation block will be added when the paper receives a public identifier.
