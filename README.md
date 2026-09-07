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

Action-prior representation sharing is a separate setting, selected at training time with `--representation_mode`:

| Representation mode | Q encoder | V encoder | Policy encoder |
|---|---|---|---|
| `all` | frozen LeWM | frozen LeWM | frozen LeWM |
| `pi` | trainable pixel | trainable pixel | frozen LeWM |
| `v` | frozen LeWM | frozen LeWM | trainable pixel |

Here `v` names the critic/value side of the action prior, so both Q and V use the frozen LeWM representation. It is not a mode in which only the scalar V network is shared. The four-task LeWM++ paper configuration uses `all`; the loader and evaluator also restore `pi` and `v` checkpoints and route policy inputs through the correct encoder automatically. Evaluation defaults to `--action-prior-representation-mode=all`; set that argument explicitly to `pi` or `v` when evaluating the corresponding checkpoint. Checkpoint metadata records the mode and all three sharing flags, and evaluation rejects an unexpected mode or inconsistent metadata.

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

Paper-facing launchers live in `experiments/`. Each Bash file is one named experiment, contains its complete hyperparameter list, launches the four tasks concurrently on GPUs 0--3, and runs that task's evaluation seeds sequentially on the same GPU. Activate `.venv`, or invoke the scripts through `uv run bash`.

## Data and path configuration

The four tasks are Cube, PushT, Reacher, and TwoRoom. Each task needs an evaluation HDF5 file, a JPEG-backed Lance table, a frozen LeWM checkpoint, an Action-Prior-Chunk checkpoint, and the relevant generator checkpoints.

```bash
cp configs/lewmpp_paths.example.env configs/lewmpp_paths.env
source configs/lewmpp_paths.env
```

`configs/lewmpp_paths.env` is ignored by Git. Checkpoints are not committed.

Only three common settings are needed: `LEWM_DATA_ROOT`, `EXPERIMENT_ROOT`, and `PYTHON_BIN`. Launchers derive all generated paths from them. Training outputs are written below `$EXPERIMENT_ROOT/open-source-retrain/`, evaluation results below `$EXPERIMENT_ROOT/evals/lewm-4tasks/`, and precomputed latent datasets beside the source data under `$LEWM_DATA_ROOT/lewm-latents/`. The remaining entries in the file identify task-specific pretrained checkpoints and cannot be inferred from a common directory when using the released paper artifacts.

The horizon-to-family mapping is strict:

| Evaluation horizon | Family | Required `goal_sampling` | Required `max_goal_steps` |
|---|---|---|---:|
| H25 | `goalmax25` | `uniform_distance_first_aligned_future_same_trajectory_stride_5_max_25` | `25` |
| H50/H75/H100 | `general_uniform_future` | `hiql_uniform_future_same_trajectory` | unset / `null` |

Every generator-based evaluation script performs a validation-only pass before allocating an environment. It reads the `config.json` adjacent to each generator checkpoint and aborts on a family, sampling protocol, maximum-goal distance, architecture, data, or checkpoint mismatch. Output paths contain experiment group, protocol family, generator type, horizon, variant, prior mode, seed, and task.

Launchers refuse to overwrite an existing `result.json`, propagate a failure from any of the four task processes, and return nonzero when a matrix is incomplete. Move an old experiment directory before intentionally rerunning the same setting.

## Training experiments

The training pipeline is explicit and ordered:

```bash
bash experiments/train_lewm_4tasks.sh
bash experiments/precompute_lewm_latents_4tasks.sh
bash experiments/train_action-prior-chunk_4tasks.sh
bash experiments/train_subgoal_latent_path_flow_goalmax25_4tasks.sh
bash experiments/train_subgoal_latent_path_flow_general_uniform_future_4tasks.sh
```

`train_action-prior-chunk_4tasks.sh` directly records the release settings: 100,000 updates, batch size 256, seed 777, learning rate `3e-4`, discount 0.99, expectile 0.9, target-update rate 0.005, action chunk 5, temperature 3.0, no image augmentation, a 5% validation split, and `all` representation sharing.

There is one fixed training Bash for every subgoal-generator choice and family:

```text
train_subgoal_mlp_goalmax25_4tasks.sh
train_subgoal_endpoint_flow_goalmax25_4tasks.sh
train_subgoal_latent_path_flow_goalmax25_4tasks.sh
train_subgoal_mlp_general_uniform_future_4tasks.sh
train_subgoal_endpoint_flow_general_uniform_future_4tasks.sh
train_subgoal_latent_path_flow_general_uniform_future_4tasks.sh
```

The flow scripts use 16 Euler steps. `goalmax25` produces `goal_sampling=uniform_distance_first_aligned_future_same_trajectory_stride_5_max_25` with `max_goal_steps=25`; `general_uniform_future` produces `goal_sampling=hiql_uniform_future_same_trajectory` with no finite maximum.

## Evaluation experiments

The four main LeWM++ settings are separate launchers, so H25 cannot accidentally reuse the general generator and H50--H100 cannot accidentally reuse the bounded generator:

```bash
bash experiments/eval_lewmpp_h25_4tasks.sh
bash experiments/eval_lewmpp_h50_4tasks.sh
bash experiments/eval_lewmpp_h75_4tasks.sh
bash experiments/eval_lewmpp_h100_4tasks.sh
```

The LeWM rows use the corresponding four fixed launchers `eval_lewm_baseline_h{25,50,75,100}_4tasks.sh`. The three paper ablation blocks are paired inside their launchers:

```bash
bash experiments/eval_ablation_action_prior_h25_4tasks.sh
bash experiments/eval_ablation_subgoal_path_h25_4tasks.sh
bash experiments/eval_ablation_moh_h25_4tasks.sh
```

The first two use evaluation seeds 0, 1, and 42; the MoH block uses 0, 1, and 666, matching the manuscript. Each launcher reruns its own full row and its paired ablation row on identical sampled starts.

Additional release checks are `eval_action-prior-chunk_h25_4tasks.sh`, `eval_policy_mode_anchor_h25_4tasks.sh`, and `eval_subgoal_generators_h25_4tasks.sh`. Together with the action-prior ablation, these cover direct Action-Prior-Chunk, `policy_mode_anchor`, MLP/Endpoint Flow/LatentPath Flow use, and zero initialization. The ordinary main scripts use `policy_mode`.

All LeWM++ launchers write the planner settings directly: 300 CEM candidates, 5 iterations, 30 elites, planner horizon `P=2`, receding horizon `R=1`, action chunk `c=5`, and 16 flow steps. The action policy is always conditioned on the original final goal.

Aggregate completed JSON files without pooling groups, families, architectures, action-prior initialization modes, or action-prior representation modes:

```bash
uv run python impls/aggregate_lewmpp_results.py \
  --results-root "$EXPERIMENT_ROOT/evals/lewm-4tasks" \
  --output "$EXPERIMENT_ROOT/evals/lewm-4tasks/summary.csv"
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
for script in experiments/*.sh; do bash -n "$script"; done
```

Tests cover the three generator shapes, family validation, consecutive-frame history, zero/policy/policy-anchor initialization, final-goal-only policy conditioning, MoH/terminal scoring, direct Action-Prior-Chunk execution, and the reduced release surface.

## Repository layout

```text
experiments/                One complete Bash launcher per experiment
configs/lewmpp_paths.env    Machine-local paths only (ignored by Git)
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
