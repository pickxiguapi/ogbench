# Archived training scripts

The eight `20260809_train_lewm_*_{gciql,hiql}_s100k.sh` files were Server 23
scripts that differed only by task and algorithm. They were consolidated on
2026-08-18 into:

- `scripts/train/20260818_train_s23_lewm_gciql_task_s100k.sh`
- `scripts/train/20260818_train_s23_lewm_hiql_task_s100k.sh`

Pass one task argument: `cube`, `pusht`, `reacher`, or `tworoom`. The archived
files are retained only to reproduce the exact historical commands and are not
current launch entry points.

The four `20260812_train_s23_lewm_jax_impala_*_e10.sh` files were likewise
consolidated into `scripts/train/20260819_train_s23_lewm_jax_impala_task_e10.sh`.
