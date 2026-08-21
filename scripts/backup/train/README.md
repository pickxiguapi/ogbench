# Archived training scripts

本目录保存已经退出活跃入口的历史训练 Bash，只用于追溯旧实验命令，不得用于启动新实验。归档脚本均取消可执行权限。

2026-08-21 对 `scripts/train/` 做了统一整理：同一服务器、算法和超参数下仅因任务不同而拆分的脚本，被合并为一个带任务数组的正式 Bash；旧的单任务脚本、launcher、queue、retry、legacy 版本以及 Server 11、Server 23、Server 7002 的历史专用入口全部迁入本目录。此次整理只发生在本地，没有同步 GitHub 或实验服务器。

当前正式入口位于 `scripts/train/20260821_train_yb_*.sh`，分为以下实验族：

- LeWM 四任务：GCIQL、GCIQL-Chunk DDPG+BC、GCIQL-Chunk AWR、稳定化 HIQL、HIQL-Chunk-GCIQL-Low-AWR、HIQL-Chunk-Share-V。
- LeWM-JAX 四任务：seed3072 主配置，以及 seed0/seed42 八运行配置。
- Visual Play/Noisy：GCIQL-Chunk DDPG+BC、GCIQL-Chunk AWR、官方 HIQL、HIQL-Chunk Two-V、HIQL-Chunk-GCIQL-Low-AWR。
- Visual Play 世界模型：LeWM-JAX frameskip/action-block 5、batch size 512、500k steps。

归档内容包括：

- `20260809_*`：Server 23 的早期 LeWM GCIQL/HIQL 单任务脚本。
- `20260811_*`：英博云旧 checkout 下的 LeWM 四任务单任务策略和 LeWM-JAX e100 脚本。
- `20260812_*`、`20260813_*`：Server 11 AWR launcher/retry，以及英博云 Visual EXP-015/016 和长时程导航脚本。
- `20260814_*`：Server 23 官方 HIQL 与 Server 7002 Noisy HIQL 脚本。
- `20260818_*`：旧的 HIQL/HIQL-Chunk 单任务+launcher、Server 23 四任务基线及早期英博云 LeWM-JAX 入口。
- `20260819_*`：Server 11 GCHIQL、Server 23/7002 LeWM-JAX、旧版英博云八任务 AWR/HIQL-Chunk 脚本。
- 已归档的 `20260821_*`：Server 11 Share-V 单任务+launcher，以及英博云 LeWM-JAX seed0/seed42 单任务+launcher；它们已由正式合并 Bash 覆盖。

若要复现某个历史实验，应先根据文件名确认原服务器，并检查脚本中的旧代码、数据、环境和输出路径；不要把这些脚本复制回 `scripts/train/` 直接运行。
