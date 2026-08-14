# Release audit for `1.2.1+pickxiguapi.1`

This document records the pre-release compatibility and integration checks for
the local OGBench fork. The release branch is `main`; `master` remains an
upstream mirror.

## Release scope

- Preserve the six upstream agents and all upstream OGBench environments.
- Add the continuous-action `GCIQL-Chunk` agent with an explicit
  `action_horizon` capability.
- Add the JAX LeWM + IMPALA-small training implementation and its independent
  CEM evaluator.
- Bundle Cube Single, PushT, TwoRoom, and Reacher directly in
  `ogbench.lewm_envs`.
- Remove the runtime dependency on a Stable World Model checkout, package,
  Python environment, or subprocess bridge. Existing data directory names are
  storage locations only.
- Remove the unreviewed HIQL-Chunk experiment.

## Built-in environment IDs

| Task | Gymnasium ID |
| --- | --- |
| Cube Single | `ogbench-lewm/CubeSingle-v0` |
| PushT | `ogbench-lewm/PushT-v1` |
| TwoRoom | `ogbench-lewm/TwoRoom-v1` |
| Reacher | `ogbench-lewm/Reacher-v0` |

## OGBench checkpoint evaluation

The following release-audit evaluations ran on Yingbo Cloud with real step-100k
checkpoints, seed 42, 50 episodes, goal offset 25, and an evaluation budget of
50. Every run loaded its checkpoint and completed against the built-in OGBench
environment implementation.

| Agent | Cube | PushT | TwoRoom | Reacher |
| --- | ---: | ---: | ---: | ---: |
| GCIQL | 88% | 90% | 100% | 96% |
| GCIQL-Chunk (`k=5`) | 96% | 82% | 100% | 90% |
| HIQL | 92% | 82% | 100% | 100% |

The successful dashboard runs use the `EXP-019-YB-*` run IDs. The initial
`EXP-019-YB-CUBE-GCIQLCHUNK-R01` attempt was rejected by the recorder before
evaluation because it incorrectly used `--eval-only` without an existing
training run. The corrected `R02` completed successfully; the failed record is
retained for auditability.

After replacing the evaluator's per-transition Python lookup dictionary with
the HDF5 `ep_offset`/`ep_len` index, all four datasets produced exactly the same
fixed evaluation samples. `EXP-019-YB-TWOROOM-GCIQL-R02` then repeated the
50-episode checkpoint evaluation and retained 100% success.

## JAX LeWM evaluation

The LeWM-JAX + IMPALA-small implementation has already produced effective
training and evaluation results on Server 23. In addition, the migration audit
loaded the existing checkpoint for each of the four tasks and completed a JAX
CEM rollout in the corresponding built-in OGBench environment. These checks are
recorded as `EXP-018-S23-JAXCEM-*`.

## Automated checks

- GCIQL-Chunk agent and dataset tests: 11 passed.
- Chunk utility tests: 4 passed.
- LeWM-JAX IMPALA forward/loss and default-configuration tests: passed.
- Built-in environment evaluation tests, including explicit action horizons,
  old-agent compatibility, and HDF5 offset indexing: 5 passed.
- HDF5-to-Lance conversion round trip: passed.
- Ruff, Python compilation, Bash syntax, lock-file validation, and
  `git diff --check`: passed.
- Source and wheel builds succeeded; the wheel contains the complete
  `ogbench.lewm_envs` package and its provenance notice.

## Intentional duplication

Runtime environment, policy, and evaluator implementations are not duplicated.
Experiment Bash files intentionally repeat complete configurations because each
one is an immutable run snapshot referenced by `experiment-dashboard`.
Historical failed/retry scripts are marked `legacy` and are not supported as
new experiment launchers.
