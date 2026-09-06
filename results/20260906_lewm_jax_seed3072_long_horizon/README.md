# LeWM-JAX seed-3072 long-horizon evaluation

Pure LeWM-JAX/CEM evaluation on the four LeWM tasks. All four world-model
checkpoints, including PushT, use training seed 3072 and epoch 10.

- Evaluation seeds: `0, 1, 42`
- Episodes: 50 per task/horizon/evaluation-seed cell
- Goal offsets: `25, 50, 75, 100`
- Evaluation budget: `2H`
- Planner: MoH, H5/RH1, action block 5, CEM 300 samples x 30 iterations
- Policy guidance: none
- Subgoal generator: none
- Launch commit: `c63652af7b2d002b550193442044716b917a5ef2`
- Remote raw results: `/data-training/yyf/ogbench-lewm-policy-runs/evals/lewm-4tasks/20260906_lewm_jax_allseed3072_generator_none_moh_cem300x30_h5_rh1_ep50/`

`summary.tsv` reports the task means and the four-task macro mean. Its final
column is the sample standard deviation of the four-task macro score across
the three evaluation seeds. `aggregate.tsv` contains task-wise mean and sample
standard deviation. `per_seed.tsv` contains all 48 cell scores.

The completed run passed a protocol audit over all 48 result JSON files:
every checkpoint path contains `seed3072`, `use_subgoal=false`, controller is
`lewm_cem`, policy guidance is `none`, CEM is 300x30, and budget is `2H`.

