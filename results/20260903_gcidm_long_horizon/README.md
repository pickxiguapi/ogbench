# GC-IDM long-horizon evaluation on node4

GC-IDM was evaluated on the four frozen-LeWM tasks with goal offset
`H in {25, 50, 75, 100}` and environment budget `B=2H`. Each cell uses 50
episodes and evaluation seed 42.

| Task | H25 | H50 | H75 | H100 |
|---|---:|---:|---:|---:|
| TwoRoom | 100.0 | 100.0 | 100.0 | 100.0 |
| Reacher | 100.0 | 100.0 | 98.0 | 100.0 |
| PushT | 92.0 | 68.0 | 28.0 | 40.0 |
| Cube | 100.0 | 98.0 | 86.0 | 84.0 |

## Provenance

- GC-IDM source: official commit `48c45b1cb2b34dd2c1c61d222c8309de567fde55`.
- Launcher: `exp/eval/lewm_4tasks/20260903_run_node4_gcidm_h25_50_75_100.sh`.
- GitHub launcher fix: `ab212f8`.
- Remote output: `/data-training/yyf/outputs/latent-geometry/eval-long-horizon/gcidm_official_48c45b1c_ep50_seed42_budget2h`.
- Completion evidence: the remote output contains `DONE`, `summary.tsv`, and 16 per-cell logs.
- Environment: `/data-training/yyf/envs/latent-geometry/bin/python`, PyTorch 2.9.1+cu128, NVIDIA A800-SXM4-80GB.

All four released GC-IDM checkpoints have `max_horizon=50`. At H75 and H100,
the goal is therefore outside the training horizon and the policy clips its
remaining-horizon conditioning to 50, as implemented by the official evaluator.

## Comparison warning

These values are a new single-seed run under the LeWM++ `B=2H` protocol, not
the numbers reported in the GC-IDM paper. The GC-IDM paper varies goal offset
over `{5, 10, 15, 25, 35, 50}` with a fixed evaluation budget of 50. SAGE uses
three 50-query manifests and uses `B=2H` for PushT but `B=H` for Cube. A joint
table must preserve these protocol labels and must not present all rows as a
strictly matched comparison.
