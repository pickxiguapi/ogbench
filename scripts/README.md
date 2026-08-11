# Experiment Scripts

本目录按 `AGENTS.md` 管理：实验 Bash 使用八位日期前缀，一个文件保存一次明确实验的完整配置。

- `train/YYYYMMDD_train_*.sh`、`eval/YYYYMMDD_eval_*.sh`：可追溯的单次实验配置快照。
- `YYYYMMDD_prepare_*.sh`：固定的数据准备步骤。
- `YYYYMMDD_test_*.sh`：代码测试，不属于实验运行。
- `train/YYYYMMDD_legacy_*.sh`、`eval/YYYYMMDD_legacy_*.sh`：旧的批量、参数化或内部启动 `tmux` 的历史脚本，仅用于追溯，禁止直接复用或作为新脚本模板。
- `convert_lewm_hdf5_to_lance.py`：数据转换工具，不直接启动实验，因此不要求日期前缀。

新实验不要修改旧脚本。复制所需命令到当天的新文件，写死 task、GPU、checkpoint、seed、超参数、输出目录和日志路径，并通过 `experiment-dashboard/scripts/recorded_run.sh` 启动。
