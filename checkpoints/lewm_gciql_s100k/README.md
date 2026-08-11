# LeWM GCIQL 100k Checkpoints

英博云统一目录：`/root/data/yyf/ogbench/checkpoints/lewm_gciql_s100k/`

| Task | Checkpoint | Reference result |
| --- | --- | --- |
| TwoRoom | `tworoom/params_100000.pkl` | 50/50, 100% |
| Reacher | `reacher/params_100000.pkl` | 48/50, 96% |
| PushT | `pusht/params_100000.pkl` | 45/50, 90% |
| Cube Single | `cube_single/params_100000.pkl` | 44/50, 88% |

每个任务目录同时保存对应的 `flags.json`。服务器上 checkpoint 使用硬链接整理，不额外占用约 2.7 GB 空间；即使原始 run 目录被移动或删除，只要本目录未删除，checkpoint 数据仍然存在。

参考评测协议：seed 42、50 episodes、goal offset 25、evaluation budget 50。
