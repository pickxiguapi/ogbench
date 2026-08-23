# Change log

## Unreleased (2026-08-23)

- Consolidate method training into `train_lewm_jax.py` and `train_gciql_chunk.py`.
- Add four explicit GCIQL-Chunk representation modes: `independent`, `pi`, `qv`, and `all`.
- Apply optional image augmentation before both pixel and frozen-LeWM encoding so representation comparisons use the same observations.
- Route native-Q guidance through the public policy `score_actions()` interface, including frozen LeWM Q representations.
- Validate shared-Q planner compatibility with the normalized LeWM checkpoint path only; no checkpoint hashing.
- Split final evaluation by protocol into `eval_lewm_4tasks.py` and `eval_ogbench_env_8tasks.py`.
- Move retired implementations and experiment Bash files to the repository-level `backup/` directory; active Bash files now live in `exp/`.
- Add suite-level reproduction wrappers for the complete four-representation training matrix and policy/LeWM/guided/native-Q evaluation matrix.

## ogbench 1.2.1+pickxiguapi.1 (2026-08-14)

This is the first separately versioned release of the `pickxiguapi/ogbench`
fork. It is based on upstream OGBench 1.2.1 and keeps `main` as the fork's
release branch; `master` remains an upstream mirror.

### Added

- Add GCIQL-Chunk for continuous-control tasks. Its critic uses exact
  discounted chunk returns and `gamma ** chunk_size` bootstrapping, while its
  actor emits a flattened fixed-length action sequence.
- Add the explicit `agent.action_horizon` capability. Public evaluation uses
  this capability instead of inferring action chunks from an unrelated
  `config.chunk_size` field, preserving atomic actions for all upstream agents.
- Add Lance-backed LeWM datasets and resumable HDF5-to-Lance conversion with
  JPEG image storage and finite terminal actions.
- Add JAX LeWM training with the OGBench IMPALA-small encoder in
  `impls/train_lewm_jax.py`.
- Add two independent, clearly named evaluation entry points:
  - `impls/eval_lewm_jax_cem.py` loads a JAX LeWM checkpoint and performs CEM
    world-model planning.
  - `impls/eval_ogbench_agent_lewm_envs.py` loads an OGBench GCIQL,
    GCIQL-Chunk, or HIQL checkpoint and evaluates the policy directly.
- Bundle four evaluation environments directly in `ogbench.lewm_envs`:
  - `ogbench-lewm/CubeSingle-v0`
  - `ogbench-lewm/PushT-v1`
  - `ogbench-lewm/TwoRoom-v1`
  - `ogbench-lewm/Reacher-v0`
- Add a shared NumPy/JAX dataset-goal protocol for HDF5 start/goal restoration,
  action normalization, rollout, success accounting, and optional video output.
- Add environment provenance and MIT attribution in
  `ogbench/lewm_envs/NOTICE.md`.
- Add `RELEASE_AUDIT.md` with the compatibility boundary, test inventory, and
  real-checkpoint validation matrix.

### Changed

- Mark the fork as version `1.2.1+pickxiguapi.1` to avoid confusion with the
  upstream `1.2.1` package.
- Support Python 3.10 through 3.12 (`>=3.10,<3.13`).
- Pin the validated runtime stack to JAX/JAXlib 0.4.33, Flax 0.8.5,
  Pygame 2.6.1, Pymunk 7.1.0, and OpenCV Headless 4.11.0.86.
- Make Lance dataset training skip unsupported online evaluation in `main.py`.
  Checkpoint evaluation is performed explicitly through
  `eval_ogbench_agent_lewm_envs.py`.
- Replace the HDF5 evaluator's per-transition Python lookup dictionary with the
  dataset's `ep_offset`/`ep_len` index. This preserves the fixed-seed evaluation
  sample sequence while avoiding a multi-million-entry dictionary.
- Reimplement TwoRoom state handling with NumPy so the built-in environment
  suite does not require Torch.
- Rename the former ambiguous entry points:
  - `train_lewm.py` -> `train_lewm_jax.py`
  - JAX model evaluation -> `eval_lewm_jax_cem.py`
  - OGBench policy evaluation -> `eval_ogbench_agent_lewm_envs.py`
- Update setup, conversion, training, and evaluation scripts to use one OGBench
  checkout and one OGBench Python environment.

### Removed

- Remove the runtime dependency on a Stable World Model checkout, Python
  package, virtual environment, or subprocess bridge. Existing directories
  named `stablewm-data` are data locations only.
- Remove the parallel Torch-based evaluation bridge.
- Remove unused Shapely helpers from the migrated PushT rendering code.
- Remove the experimental HIQL-Chunk agent and dataset pending a separate
  algorithm review.
- Remove stale ViT documentation; JAX LeWM now documents the single supported
  IMPALA-small implementation.

### Validation

- Evaluate real step-100k GCIQL, GCIQL-Chunk, and HIQL checkpoints on all four
  built-in environments: 12/12 runs completed, with success rates from 82% to
  100%. Runs are recorded under `EXP-019-YB-*`.
- Load existing JAX LeWM checkpoints and complete CEM rollouts on all four
  built-in environments; runs are recorded under `EXP-018-S23-JAXCEM-*`.
- Pass the GCIQL-Chunk, chunk-return, LeWM-JAX, HDF5 indexing, explicit action
  horizon, and old-agent evaluation compatibility tests.
- Pass Ruff on changed Python files, Python compilation, Bash syntax checks,
  lock-file validation, wheel construction, and wheel-content inspection.

See [RELEASE_AUDIT.md](RELEASE_AUDIT.md) for the exact validation matrix and
known audit-only failed recorder attempt.

## ogbench 1.2.1 (2026-01-14)
- Make it compatible with the latest version of `numpy` (2.0.0+).

## ogbench 1.2.0 (2025-10-20)
- Make `singletask` environments compute rewards based on `s` instead of `s'` for an `(s, a, s')` tuple.
See [this discussion](README.md/#caveats).

## ogbench 1.1.5 (2025-07-02)
- Make locomotion environments compatible with the headless mode.

## ogbench 1.1.4 (2025-06-17)
- Fix the black rendering issue in locomotion environments.

## ogbench 1.1.3 (2025-06-03)
- Add the `cube-octuple` task.

## ogbench 1.1.2 (2025-03-30)
- Improve compatibility with `gymnasium`.

## ogbench 1.1.1 (2025-03-02)
- Make it compatible with the latest version of `gymnasium` (1.1.0).

## ogbench 1.1.0 (2025-02-13)
- Added `-singletask` environments for standard (i.e., non-goal-conditioned) offline RL.
- Added `-oraclerep` environments for offline goal-conditioned RL with oracle goal representations.

## ogbench 1.0.1 (2024-10-28)
- Fixed a bug in the reward function of manipulation tasks.

## ogbench 1.0.0 (2024-10-25)
- Initial release.
