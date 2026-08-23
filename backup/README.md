# Backup

This directory contains retired experiment launchers and superseded implementations. It is retained only for historical traceability and must not be used to start new experiments.

- `scripts/preexisting/`: the repository's earlier backup collection.
- `scripts/retired_20260823/`: all train, eval, and setup Bash entrypoints retired when the final LeWM-JAX + GCIQL-Chunk workflow was consolidated.
- `impls/train/`: superseded selective-sharing and standalone shared-Q training entrypoints.
- `impls/agents/`: the superseded standalone shared-Q/V evaluator.
- `impls/eval/`: pre-consolidation exploratory evaluation implementations.
- `impls/tests/`: tests tied only to retired implementations.

The active experiment Bash files live in `exp/`; active Python entrypoints are documented in `METHOD.md` and `AGENTS.md`.
