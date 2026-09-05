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
