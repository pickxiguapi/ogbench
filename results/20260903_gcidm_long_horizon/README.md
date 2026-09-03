# GC-IDM long-horizon evaluation on node4

GC-IDM was evaluated on the four frozen-LeWM tasks with goal offset
`H in {25, 50, 75, 100}` and environment budget `B=2H`. Each run uses 50
episodes per cell. The table reports mean +/- sample standard deviation over
the evaluation-manifest seeds `{32, 42, 52}`.

| Task | H25 | H50 | H75 | H100 |
|---|---:|---:|---:|---:|
| TwoRoom | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
| Reacher | 100.0 +/- 0.0 | 99.3 +/- 1.2 | 98.0 +/- 0.0 | 99.3 +/- 1.2 |
| PushT | 88.0 +/- 4.0 | 71.3 +/- 4.2 | 48.7 +/- 17.9 | 39.3 +/- 1.2 |
| Cube | 99.3 +/- 1.2 | 92.0 +/- 6.0 | 76.7 +/- 9.0 | 80.7 +/- 5.8 |
| Macro average | 96.8 +/- 1.3 | 90.7 +/- 0.8 | 80.8 +/- 2.6 | 79.8 +/- 1.6 |

## Provenance

- GC-IDM source: official commit `48c45b1cb2b34dd2c1c61d222c8309de567fde55`.
- Launcher: `exp/eval/lewm_4tasks/20260903_run_node4_gcidm_h25_50_75_100.sh`.
- Remote outputs: `/data-training/yyf/outputs/latent-geometry/eval-long-horizon/gcidm_official_48c45b1c_ep50_seed{32,42,52}_budget2h`.
- Completion evidence: all three output directories contain `DONE`, `summary.tsv`, and 16 per-cell logs (48 completed cells total).
- Environment: `/data-training/yyf/envs/latent-geometry/bin/python`, PyTorch 2.9.1+cu128, NVIDIA A800-SXM4-80GB.
- `per_seed.tsv` contains all 48 raw cell values; `aggregate.tsv` contains the
  mean and sample standard deviation (`ddof=1`). The legacy `summary.tsv`
  preserves the original seed-42 run.

All four released GC-IDM checkpoints have `max_horizon=50`. At H75 and H100,
the goal is therefore outside the training horizon and the policy clips its
remaining-horizon conditioning to 50, as implemented by the official evaluator.

## Comparison warning

These values are new multi-manifest runs under the LeWM++ `B=2H` protocol, not
the numbers reported in the GC-IDM paper. The GC-IDM paper varies goal offset
over `{5, 10, 15, 25, 35, 50}` with a fixed evaluation budget of 50. SAGE uses
three 50-query manifests and uses `B=2H` for PushT but `B=H` for Cube. A joint
table must preserve these protocol labels and must not present all rows as a
strictly matched comparison.

The three seeds change evaluation start-goal manifests, not GC-IDM training.
All runs use the same released GC-IDM and frozen-LeWM checkpoints.
