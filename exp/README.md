# Active experiments

Only Bash files in this directory are active experiment entrypoints.

Each executor resolves `OGBENCH_ROOT` from its own location and sources that
checkout's `scripts/client_env.sh`; the repository may therefore live at a
different absolute path on each server.

Formal representation comparisons default to `P_AUG=0.0`. On LeWM-4Tasks,
the independent GCIQL-Chunk baseline has its own Bash; the shared policy Bash
requires `REPRESENTATION_MODE=pi`, `qv`, or `all`. Evaluation accepts all four
modes.

Canonical order:

1. Train `independent` GCIQL-Chunk and LeWM-JAX (these two are independent and may run in parallel).
2. After the LeWM checkpoint exists, train `pi`, `qv`, and `all` from that frozen checkpoint.
3. Evaluate `policy`, `lewm`, `guided`, and `native_q` with the matching suite Bash.

Train LeWM and GCIQL-Chunk with their separate executor Bash files. Evaluation
matrices retain convenience wrappers:

```bash
bash exp/eval/lewm_4tasks/20260823_reproduce_yb_lewm_4tasks_main_matrix.sh

bash exp/eval/ogbench_env_8tasks/20260823_reproduce_node2_ogbench_env_8tasks_main_matrix.sh
```

Evaluation wrappers expose `RUN_POLICY`, `RUN_LEWM`, `RUN_GUIDED`, and
`RUN_NATIVE_Q`. The underlying executor parameters use separate
`POLICY_*` and `LEWM_*` names, so customized settings propagate consistently
from training through checkpoint lookup and evaluation.

LeWM-4Tasks example:

```bash
bash exp/train/20260823_train_yb_gciql_chunk_4tasks_independent.sh
bash exp/train/20260823_train_yb_lewm_4tasks.sh
export LEWM_CUBE_CHECKPOINT=/path/to/cube.msgpack
export LEWM_PUSHT_CHECKPOINT=/path/to/pusht.msgpack
export LEWM_REACHER_CHECKPOINT=/path/to/reacher.msgpack
export LEWM_TWOROOM_CHECKPOINT=/path/to/tworoom.msgpack
for mode in pi qv all; do
  REPRESENTATION_MODE="$mode" bash exp/train/20260823_train_yb_gciql_chunk_4tasks.sh
done
MODE=guided REPRESENTATION_MODE=pi \
  bash exp/eval/lewm_4tasks/20260823_eval_yb_lewm_4tasks.sh
```

Use the two `node2` Bash files analogously for OGBench-Env-8Tasks. The policy
and evaluation Bash files derive checkpoint directories from the same exposed
step, seed, batch-size, augmentation, and representation-mode variables.
Independent policy training neither accepts nor resolves a LeWM checkpoint.
For `pi`, `qv`, and `all`, the LeWM-4Tasks policy Bash requires one explicit
checkpoint path per task and only loads those frozen checkpoints. LeWM training
metadata such as its batch size and seed is not a policy-training parameter.
LeWM itself is trained exclusively by `train_lewm_jax.py` through its own Bash.
Policy experiment names use the compact shared format
`gc{4|8}_${task}_${mode}_n${steps}_b${batch}_a${p_aug}_sd${seed}` (with
`independent` shortened to `ind`). Both training and evaluation reject names
that are 64 characters or longer, so customized values cannot create an
invalid checkpoint name silently.
Evaluation directories include the main checkpoint/CEM settings; set
`EVAL_TAG` explicitly when running another variant that should remain separate.

Do not launch files under `backup/`.
