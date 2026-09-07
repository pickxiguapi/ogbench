# LeWM++

LeWM++ is a closed-loop latent-space planner for long-distance image-goal control. It combines a frozen LeWM world model with three independently switchable components:

1. a subgoal generator that maps observation history and the remote final goal to a reachable latent target;
2. a final-goal-conditioned Action-Prior-Chunk that initializes CEM;
3. min-over-horizon (MoH) trajectory scoring.

This release branch reproduces the LeWM four-task experiments at goal offsets H25, H50, H75, and H100. It also provides the three paper ablations, standalone LeWM and Action-Prior-Chunk baselines, three subgoal-generator architectures, and two generator sampling families.

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
| `action_prior_chunk` | off | direct execution | n/a | standalone Action-Prior-Chunk |

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

The two public launchers are deliberately small and live at the repository root:

```text
train.sh   Read one training config and run its Python entrypoint.
eval.sh    Read one evaluation config, validate it, and run evaluation.
```

Activate `.venv`, or invoke them through `uv run bash`.

## Data and path configuration

The four tasks are Cube, PushT, Reacher, and TwoRoom. Each task needs an evaluation HDF5 file, a JPEG-backed Lance table, a frozen LeWM checkpoint, an Action-Prior-Chunk checkpoint, and the relevant generator checkpoints.

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

Before evaluation, `eval.sh` asks the evaluator to perform a validation-only pass. It reads the `config.json` adjacent to the generator checkpoint and aborts on a family, sampling protocol, maximum-goal distance, architecture, data, or checkpoint mismatch. Output paths in the provided configs contain experiment group, generator family, generator type, horizon, variant, prior mode, seed, and task, so incompatible results do not overwrite or silently pool.

## Config-driven training

Every path, hyperparameter, GPU assignment, entrypoint, and output location is written in a config. The Bash launcher contains no experiment-specific branches.

| Config template | Purpose |
|---|---|
| `configs/train_lewm.example.env` | Train the frozen LeWM |
| `configs/precompute_latents.example.env` | Build the checkpoint-bound latent cache |
| `configs/action-prior-chunk.example.env` | Train final-goal Action-Prior-Chunk |
| `configs/train_subgoal_generator.example.env` | Train MLP, Endpoint Flow, or LatentPath Flow |

Copy the desired template, fill its paths, and run it:

```bash
cp configs/train_lewm.example.env configs/train_lewm.env
bash train.sh configs/train_lewm.env
```

The same launcher runs every training stage:

```bash
bash train.sh configs/precompute_latents.env
bash train.sh configs/action-prior-chunk.env
bash train.sh configs/train_subgoal_generator.env
```

In the subgoal config, select `GENERATOR_TYPE` from `mlp`, `endpoint_flow`, and `latent_path_flow`, and select `GENERATOR_FAMILY` from `goalmax25` and `general_uniform_future`. The released flow configuration uses 16 Euler steps. The Action-Prior-Chunk config fixes chunk size 5, training seed 777, and the shared frozen LeWM representation used by Q, V, and policy.

## Config-driven evaluation

Copy the path file and one complete evaluation config:

```bash
cp configs/lewmpp_paths.example.env configs/lewmpp_paths.env
cp configs/eval_lewmpp_h25.example.env configs/eval_lewmpp_h25.env
bash eval.sh configs/eval_lewmpp_h25.env
```

The provided evaluation configs are directly runnable after their paths are filled:

| Config template | Experiment |
|---|---|
| `eval_lewmpp_h25.example.env` | Full LeWM++ with the H25 `goalmax25` generator |
| `eval_lewmpp_general.example.env` | Full LeWM++ at H50/H75/H100 with the general generator |
| `eval_no_subgoal.example.env` | LeWM++ w/o Subgoal Path |
| `eval_no_action_prior.example.env` | LeWM++ w/o Action Prior, zero initialization |
| `eval_no_moh.example.env` | LeWM++ w/o MoH, terminal cost |
| `eval_lewm_baseline.example.env` | Standalone LeWM |
| `eval_action_prior_chunk.example.env` | Standalone Action-Prior-Chunk |

Each file contains the complete command arguments rather than relying on hidden defaults. To use `policy_mode_anchor`, change the action-prior mode and its corresponding output-directory component in a copied config. To evaluate an MLP or Endpoint Flow checkpoint, change `--generator-type` and `--subgoal-generator-checkpoint` together. The Python preflight rejects inconsistent combinations.

The paper planner settings are explicit in every LeWM++ config: 300 candidates, 5 CEM iterations, 30 elites, planning horizon `P=2`, receding horizon `R=1`, action chunk `c=5`, and 16 flow steps. The policy is always conditioned on the original final goal.

To reproduce a matrix, create one config per task, horizon, and evaluation seed, then launch those config files with `eval.sh`. Keeping each run explicit makes its checkpoint family, GPU, seed, and output provenance reviewable without reading launcher logic.

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
bash -n train.sh
bash -n eval.sh
for config in configs/*.example.env; do bash -n "$config"; done
```

Tests cover the three generator shapes, family validation, consecutive-frame history, zero/policy/policy-anchor initialization, final-goal-only policy conditioning, MoH/terminal scoring, direct Action-Prior-Chunk execution, and the reduced release surface.

## Repository layout

```text
train.sh                    Config-driven training launcher
eval.sh                     Config-driven evaluation launcher and preflight
configs/                    Complete train/eval config templates
impls/action_prior_chunk.py Action-Prior-Chunk loader and direct policy
impls/action-prior-chunk.py Action-Prior-Chunk training entrypoint
impls/subgoal_generators.py MLP, Endpoint Flow, and LatentPath Flow
impls/lewm_jax/planner.py   Canonical LeWM++ controller
impls/train_*.py            LeWM, action-prior, and generator trainers
impls/eval_lewm_4tasks.py   Unified experiment entrypoint
ogbench/                    OGBench environments and data APIs
```

## License and citation

The repository retains OGBench's MIT license. Please cite OGBench and LeWorldModel when using their benchmark, environments, or model implementation. A LeWM++ citation block will be added when the paper receives a public identifier.
