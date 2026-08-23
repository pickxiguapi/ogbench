# Active experiments

Only Bash files in this directory are active experiment entrypoints.

Each executor resolves `OGBENCH_ROOT` from its own location and sources that
checkout's `scripts/client_env.sh`; the repository may therefore live at a
different absolute path on each server.

For each policy training or evaluation Bash, set `REPRESENTATION_MODE` to one of `independent`, `pi`, `qv`, or `all`. Formal representation comparisons default to `P_AUG=0.0`.

Canonical order:

1. Train `independent` GCIQL-Chunk and LeWM-JAX (these two are independent and may run in parallel).
2. After the LeWM checkpoint exists, train `pi`, `qv`, and `all` from that frozen checkpoint.
3. Evaluate `policy`, `lewm`, `guided`, and `native_q` with the matching suite Bash.

The simplest full reproductions are:

```bash
bash exp/train/20260823_reproduce_yb_lewm_4tasks_main_matrix.sh
bash exp/eval/lewm_4tasks/20260823_reproduce_yb_lewm_4tasks_main_matrix.sh

bash exp/train/20260823_reproduce_node2_ogbench_env_8tasks_main_matrix.sh
bash exp/eval/ogbench_env_8tasks/20260823_reproduce_node2_ogbench_env_8tasks_main_matrix.sh
```

Training wrappers expose `RUN_INDEPENDENT`, `RUN_LEWM`, and `RUN_SHARED` phase
switches. Evaluation wrappers expose `RUN_POLICY`, `RUN_LEWM`, `RUN_GUIDED`,
and `RUN_NATIVE_Q`. The underlying executor parameters use separate
`POLICY_*` and `LEWM_*` names, so customized settings propagate consistently
from training through checkpoint lookup and evaluation.

LeWM-4Tasks example:

```bash
REPRESENTATION_MODE=independent bash exp/train/20260823_train_yb_gciql_chunk_4tasks.sh
bash exp/train/20260823_train_yb_lewm_4tasks.sh
for mode in pi qv all; do
  REPRESENTATION_MODE="$mode" bash exp/train/20260823_train_yb_gciql_chunk_4tasks.sh
done
MODE=guided REPRESENTATION_MODE=pi \
  bash exp/eval/lewm_4tasks/20260823_eval_yb_lewm_4tasks.sh
```

Use the two `node2` Bash files analogously for OGBench-Env-8Tasks. The policy
and evaluation Bash files derive checkpoint directories from the same exposed
step, seed, batch-size, augmentation, and representation-mode variables.
Policy experiment names use the compact shared format
`gc{4|8}_${task}_${mode}_n${steps}_b${batch}_a${p_aug}_sd${seed}` (with
`independent` shortened to `ind`). Both training and evaluation reject names
that are 64 characters or longer, so customized values cannot create an
invalid checkpoint name silently.
Evaluation directories include the main checkpoint/CEM settings; set
`EVAL_TAG` explicitly when running another variant that should remain separate.

Do not launch files under `backup/`.
