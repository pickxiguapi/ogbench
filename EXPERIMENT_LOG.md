# Experiment Log

## 2026-09-06 — LeWM++ w/o MoH, H25

**Question.** Does min-over-horizon improve H25 success when all other components of canonical LeWM++ are fixed?

**Single change.** Replace `cem_cost_mode=moh` with terminal/last-state cost (`cem_cost_mode=last`).

**Fixed protocol.** H25 goalmax25 LatentPathFlow K10; FlowPath `num_samples=1`; Policy mode with shared-all GCIQL-Chunk-AWR seed777 and final-goal conditioning; Cube/Reacher/TwoRoom LeWM seed3072 and PushT LeWM seed666; H2/RH1/action-block5; CEM300x5/top-k30; 50 episodes; evaluation seeds 0/1/666.

| Cost | Eval seed | TwoRoom | Reacher | PushT | OGBench-Cube | Average |
|---|---:|---:|---:|---:|---:|---:|
| Terminal | 0 | 100.00 | 100.00 | 82.00 | 92.00 | 93.50 |
| Terminal | 1 | 100.00 | 100.00 | 84.00 | 88.00 | 93.00 |
| Terminal | 666 | 100.00 | 96.00 | 90.00 | 96.00 | 95.50 |

Population mean±std:

| Setting | TwoRoom | Reacher | PushT | OGBench-Cube | Average |
|---|---:|---:|---:|---:|---:|
| LeWM++ w/o MoH (Terminal) | 100.00±0.00 | 98.67±1.89 | 85.33±3.40 | 92.00±3.27 | 94.00±1.08 |
| LeWM++ (MoH, matched seeds) | 100.00±0.00 | 95.33±2.49 | 96.67±0.94 | 95.33±2.49 | 96.83±0.62 |
| Terminal − MoH | +0.00 | +3.33 | −11.33 | −3.33 | −2.83 |

**Integrity.** All 12 result JSON files passed checks for seed, task, 50 episodes, H25/budget50, Policy mode, final-goal conditioning, goalmax25 checkpoint, FlowPath ns1, policy seed777, task-specific LeWM seed, and `cem.cost_mode=last`. No traceback, NaN, or failed run was found.

**Artifacts.** Results: `/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks/20260906_goalmax25_ns1_policy_mode_terminal_sd777_cem300x5_h2_rh1_g25_b50_ep50_seed{0,1,666}/`. Launcher: `exp/eval/lewm_4tasks/20260906_eval_node4_goalmax25_ns1_policy_mode_terminal_3seeds.sh`.

## 2026-09-06 — LeWM++ without policy guidance, H25

**Question.** How much of canonical H25 LeWM++ performance remains when all
policy initialization and guidance are removed and planning uses only CEM?

**Single change.** Replace shared-all GCIQL-Chunk-AWR seed777 Policy mode with
`policy_guidance=none`; do not load any policy checkpoint. The LatentPathFlow
subgoal generator remains enabled.

**Fixed protocol.** H25 goalmax25 LatentPathFlow K10; FlowPath
`num_samples=1`; Cube/Reacher/TwoRoom LeWM seed3072 and PushT LeWM seed666;
MoH; H2/RH1/action-block5; CEM300x5/top-k30; 50 episodes; evaluation seeds
0/1/42. Policy-on and policy-off variants were rerun together from GitHub main
commit `f2a0377` so the comparison changes only policy guidance.

| Variant | Eval seed | TwoRoom | Reacher | PushT | OGBench-Cube | Average |
|---|---:|---:|---:|---:|---:|---:|
| Policy mode | 0 | 100.00 | 96.00 | 96.00 | 94.00 | 96.50 |
| Policy mode | 1 | 100.00 | 100.00 | 98.00 | 96.00 | 98.50 |
| Policy mode | 42 | 100.00 | 94.00 | 94.00 | 96.00 | 96.00 |
| Pure CEM | 0 | 100.00 | 90.00 | 94.00 | 88.00 | 93.00 |
| Pure CEM | 1 | 100.00 | 92.00 | 92.00 | 84.00 | 92.00 |
| Pure CEM | 42 | 100.00 | 84.00 | 92.00 | 88.00 | 91.00 |

Population mean±std:

| Setting | TwoRoom | Reacher | PushT | OGBench-Cube | Average |
|---|---:|---:|---:|---:|---:|
| LeWM++ (Policy mode) | 100.00±0.00 | 96.67±2.49 | 96.00±1.63 | 95.33±0.94 | 97.00±1.08 |
| LeWM++ w/o policy (Pure CEM) | 100.00±0.00 | 88.67±3.40 | 92.67±0.94 | 86.67±1.89 | 92.00±0.82 |
| Policy mode − Pure CEM | +0.00 | +8.00 | +3.33 | +8.67 | +5.00 |

**Integrity.** All 24 paired result JSON files passed checks for seed, task,
50 episodes, H25/budget50, goalmax25 checkpoint, FlowPath ns1, task-specific
LeWM seed, MoH, H2/RH1, and CEM300x5. Pure CEM result files additionally
record `policy_guidance=none` and `policy_checkpoint_dir=null`. No traceback,
NaN, or failed run was found. A separate pure-CEM-only rerun scored 92.33
overall; the 0.33-point aggregate difference indicates small run-level
variation, so the simultaneous paired run above is the primary ablation.

**Artifacts.** Paired results:
`/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks/20260906_goalmax25_ns1_paired_policy_ablation/`.
Pure-CEM-only replicate:
`/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks/20260906_goalmax25_ns1_pure_cem_no_policy_moh_cem300x5_h2_rh1_g25_b50_ep50_seed{0,1,42}/`.
Launcher:
`exp/eval/lewm_4tasks/20260906_eval_node4_goalmax25_ns1_pure_cem_3seeds.sh`.
