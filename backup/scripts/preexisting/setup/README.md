# Archived setup scripts

These scripts are retained only for historical experiment reproduction. They are
not current training or evaluation entry points and should not be launched for
new experiments.

| Script | Original target | Why archived |
| --- | --- | --- |
| `20260812_prepare_s11_lewm_lance_four_tasks.sh` | Server 11 | Server-specific one-time dataset conversion |
| `20260812_setup_s11_ogbench_env.sh` | Server 11 | Server-specific one-time environment setup |
| `20260813_setup_s11_cube_eval_env.sh` | Server 11 | Server-specific one-time evaluation environment setup |
| `20260814_prepare_s23_ogbench_lewm_envs_smoke.sh` | Server 23 | Server-specific release smoke-test preparation |
| `20260814_setup_server7002_ogbench.sh` | Server 7002 | Server-specific one-time environment setup |
| `20260814_prepare_yb_ogbench_release_audit.sh` | Yingbo Cloud | One-time check for the separate `ogbench-release-audit-20260814` checkout; retained because historical dashboard records reference it |

The active Yingbo Cloud checkout is `/root/data/yyf/ogbench-new`. New setup or
preparation logic should only be added when it is required by a current,
recorded experiment.
