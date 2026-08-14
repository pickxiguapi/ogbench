# EXP-015 legacy script map

EXP-015 的前三轮尝试均已写入 `experiment-dashboard`，不能作为无用草稿删除。为防止它们被误作新实验模板，
仓库使用带 `legacy` 的名称归档；看板保存的是运行当时的旧文件名。

| 看板中的运行时文件名 | 仓库归档文件名 |
| --- | --- |
| `20260813_train_yb_visual_cube_single_gciql_chunk_k5_s500k.sh` | `20260813_legacy_train_yb_visual_cube_single_gciql_chunk_k5_s500k_exp015_r1.sh` |
| `20260813_train_yb_visual_cube_single_gciql_chunk_k5_s500k_v2.sh` | `20260813_legacy_train_yb_visual_cube_single_gciql_chunk_k5_s500k_exp015_r2.sh` |
| `20260813_train_yb_visual_cube_single_gciql_chunk_k5_s500k_v3.sh` | `20260813_legacy_train_yb_visual_cube_single_gciql_chunk_k5_s500k_exp015_r3.sh` |
| `20260813_train_yb_visual_cube_double_gciql_chunk_k5_s500k.sh` | `20260813_legacy_train_yb_visual_cube_double_gciql_chunk_k5_s500k_exp015_r1.sh` |
| `20260813_train_yb_visual_cube_double_gciql_chunk_k5_s500k_v2.sh` | `20260813_legacy_train_yb_visual_cube_double_gciql_chunk_k5_s500k_exp015_r2.sh` |
| `20260813_train_yb_visual_cube_double_gciql_chunk_k5_s500k_v3.sh` | `20260813_legacy_train_yb_visual_cube_double_gciql_chunk_k5_s500k_exp015_r3.sh` |
| `20260813_train_yb_visual_scene_gciql_chunk_k5_s500k.sh` | `20260813_legacy_train_yb_visual_scene_gciql_chunk_k5_s500k_exp015_r1.sh` |
| `20260813_train_yb_visual_scene_gciql_chunk_k5_s500k_v2.sh` | `20260813_legacy_train_yb_visual_scene_gciql_chunk_k5_s500k_exp015_r2.sh` |
| `20260813_train_yb_visual_scene_gciql_chunk_k5_s500k_v3.sh` | `20260813_legacy_train_yb_visual_scene_gciql_chunk_k5_s500k_exp015_r3.sh` |
| `20260813_train_yb_visual_puzzle_3x3_gciql_chunk_k5_s500k.sh` | `20260813_legacy_train_yb_visual_puzzle_3x3_gciql_chunk_k5_s500k_exp015_r1.sh` |
| `20260813_train_yb_visual_puzzle_3x3_gciql_chunk_k5_s500k_v2.sh` | `20260813_legacy_train_yb_visual_puzzle_3x3_gciql_chunk_k5_s500k_exp015_r2.sh` |
| `20260813_train_yb_visual_puzzle_3x3_gciql_chunk_k5_s500k_v3.sh` | `20260813_legacy_train_yb_visual_puzzle_3x3_gciql_chunk_k5_s500k_exp015_r3.sh` |

EXP-016 的 `v4` 脚本是成功运行的正式配置快照，因此保留为非 legacy 脚本，并改用语义化的
`_exp016.sh` 后缀。看板中的旧文件名与仓库名对应如下：

| 看板中的运行时文件名 | 仓库正式文件名 |
| --- | --- |
| `20260813_train_yb_visual_cube_single_gciql_chunk_k5_s500k_v4.sh` | `20260813_train_yb_visual_cube_single_gciql_chunk_k5_s500k_exp016.sh` |
| `20260813_train_yb_visual_cube_double_gciql_chunk_k5_s500k_v4.sh` | `20260813_train_yb_visual_cube_double_gciql_chunk_k5_s500k_exp016.sh` |
| `20260813_train_yb_visual_scene_gciql_chunk_k5_s500k_v4.sh` | `20260813_train_yb_visual_scene_gciql_chunk_k5_s500k_exp016.sh` |
| `20260813_train_yb_visual_puzzle_3x3_gciql_chunk_k5_s500k_v4.sh` | `20260813_train_yb_visual_puzzle_3x3_gciql_chunk_k5_s500k_exp016.sh` |

以下未直接出现在看板 command 字段中的包装脚本也改为 legacy：它们负责批量启动或在调用
`recorded_run.sh` 后聚合结果，不是可复用的单实验快照。

- `scripts/eval/20260813_legacy_launch_yb_exp016_gciql_chunk_k5_s500k_eval.sh`
- `scripts/eval/20260813_legacy_record_s11_cube_gciql_chunk_awr_eval.sh`
- `scripts/train/20260813_legacy_relaunch_s11_lewm_gciql_chunk_awr_three_tasks.sh`
- `scripts/eval/20260813_legacy_eval_yb_exp016_gciql_chunk_k5_s500k.sh`
- `scripts/eval/20260813_legacy_eval_s11_lewm_cube_gciql_chunk_awr_s100k_seed42.sh`
- `scripts/train/20260812_legacy_train_s11_lewm_gciql_chunk_awr_s100k.sh`
