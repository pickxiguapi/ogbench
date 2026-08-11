# Experiment Scripts

本目录按 `AGENTS.md` 管理：实验 Bash 使用八位日期前缀，一个文件保存一次明确实验的完整配置。

- `train/YYYYMMDD_train_*.sh`、`eval/YYYYMMDD_eval_*.sh`：可追溯的单次实验配置快照。
- `YYYYMMDD_prepare_*.sh`：固定的数据准备步骤。
- `YYYYMMDD_test_*.sh`：代码测试，不属于实验运行。
- `train/YYYYMMDD_legacy_*.sh`、`eval/YYYYMMDD_legacy_*.sh`：旧的批量、参数化或内部启动 `tmux` 的历史脚本，仅用于追溯，禁止直接复用或作为新脚本模板。
- `convert_lewm_hdf5_to_lance.py`：数据转换工具，不直接启动实验，因此不要求日期前缀。

新实验不要修改旧脚本。复制所需命令到当天的新文件，写死 task、GPU、checkpoint、seed、超参数、输出目录和日志路径，并通过 `experiment-dashboard/scripts/recorded_run.sh` 启动。

## 英博云：OGBench GCIQL 跑 LeWM 四任务

参考配置：IMPALA-small、batch 256、alpha 1、p_aug 0.5、seed 0、100k steps、W&B offline。四个任务分别使用 GPU 4/5/6/7。以下四条命令可以分别放进 tmux 执行；真实训练 Bash 自身保持前台运行，并由记录器负责 started/completed/failed 事件。

```bash
cd /root/data/yyf/experiment-dashboard

bash scripts/recorded_run.sh EXP-009 'GCIQL_ogbench_lewm_tworoom_impala_bs256_s100k_alpha1_paug0.5_seed0_wandboffline' EXP-009-R01 \
  -- bash /root/data/yyf/ogbench/scripts/train/20260811_train_yb_lewm_tworoom_gciql_s100k.sh

bash scripts/recorded_run.sh EXP-009 'GCIQL_ogbench_lewm_reacher_impala_bs256_s100k_alpha1_paug0.5_seed0_wandboffline' EXP-009-R02 \
  -- bash /root/data/yyf/ogbench/scripts/train/20260811_train_yb_lewm_reacher_gciql_s100k.sh

bash scripts/recorded_run.sh EXP-009 'GCIQL_ogbench_lewm_pusht_impala_bs256_s100k_alpha1_paug0.5_seed0_wandboffline' EXP-009-R03 \
  -- bash /root/data/yyf/ogbench/scripts/train/20260811_train_yb_lewm_pusht_gciql_s100k.sh

bash scripts/recorded_run.sh EXP-009 'GCIQL_ogbench_lewm_cube_single_impala_bs256_s100k_alpha1_paug0.5_seed0_wandboffline' EXP-009-R04 \
  -- bash /root/data/yyf/ogbench/scripts/train/20260811_train_yb_lewm_cube_gciql_s100k.sh
```

如需四卡并行，分别创建四个 tmux，然后在每个 tmux 中粘贴对应命令。不要绕过 `recorded_run.sh` 直接执行训练 Bash。

## 英博云：复用四个参考 checkpoint 评测

四个 100k checkpoint 统一整理在 `/root/data/yyf/ogbench/checkpoints/lewm_gciql_s100k/`。评测采用 seed 42、50 episodes、goal offset 25、budget 50，并复用各自已有 Run ID：

```bash
cd /root/data/yyf/experiment-dashboard

# TwoRoom：参考结果 100%
bash scripts/recorded_run.sh EXP-006 'GCIQL_ogbench_tworoom_bs256_s100k_seed0' EXP-006-R01 \
  --eval-only bash /root/data/yyf/ogbench/scripts/eval/20260811_eval_lewm_tworoom_gciql_s100k.sh

# Reacher：参考结果 96%
bash scripts/recorded_run.sh EXP-008 'GCIQL_ogbench_reacher_bs256_s100k_seed0' EXP-008-R01 \
  --eval-only bash /root/data/yyf/ogbench/scripts/eval/20260811_eval_lewm_reacher_gciql_s100k.sh

# PushT：参考结果 90%
bash scripts/recorded_run.sh EXP-002 'GCIQL_ogbench_pusht_bs256_s100k_seed0' EXP-002-R01 \
  --eval-only bash /root/data/yyf/ogbench/scripts/eval/20260811_eval_lewm_pusht_gciql_s100k.sh

# Cube Single：参考结果 88%
bash scripts/recorded_run.sh EXP-002 'GCIQL_ogbench_cube_single_bs256_s100k_seed0' EXP-002-R02 \
  --eval-only bash /root/data/yyf/ogbench/scripts/eval/20260811_eval_lewm_cube_gciql_s100k.sh
```

以后评测新训练的 checkpoint 时，复制对应评测 Bash 为当天的新文件，写死新 checkpoint 目录和新 Run ID；不要修改这四个参考快照，也不要让评测自动猜测最新 checkpoint。

## 英博云：JAX LeWM 原版四任务复现

该组实验对应 `lucas-maes/le-wm@8edfeb3` 的模型、损失与官方超参数，并以该代码实际使用的 `stable-pretraining==0.1.8` 为依赖语义基准：ViT-Tiny/14、history 3、frameskip 5、batch 128、100 epochs、AdamW 5e-5/1e-3、SIGReg 0.09（17 knots、1024 projections）、bf16。训练从 Lance 懒加载像素序列；评测使用 dataset-goal 协议和原生 JAX CEM（300 samples、30 iterations、top-30），但仍保持官方 CEM 更新规则。

每个训练与评测必须作为同一个 pipeline Run 启动，HTML 中只占一行：

```bash
cd /root/data/yyf/experiment-dashboard

bash scripts/recorded_run.sh EXP-013 'LeWMJAX_lance_cube_single_bs128_e100_seed3072_fs5_h3_sigreg009_cem300x30' EXP-013-R01 \
  --train bash /root/data/yyf/ogbench/scripts/train/20260811_train_yb_lewm_jax_cube_e100.sh \
  --eval bash /root/data/yyf/ogbench/scripts/eval/20260811_eval_yb_lewm_jax_cube_e100_cem300x30.sh

bash scripts/recorded_run.sh EXP-013 'LeWMJAX_lance_pusht_expert_bs128_e100_seed3072_fs5_h3_sigreg009_cem300x30' EXP-013-R02 \
  --train bash /root/data/yyf/ogbench/scripts/train/20260811_train_yb_lewm_jax_pusht_e100.sh \
  --eval bash /root/data/yyf/ogbench/scripts/eval/20260811_eval_yb_lewm_jax_pusht_e100_cem300x30.sh

bash scripts/recorded_run.sh EXP-013 'LeWMJAX_lance_reacher_bs128_e100_seed3072_fs5_h3_sigreg009_cem300x30' EXP-013-R03 \
  --train bash /root/data/yyf/ogbench/scripts/train/20260811_train_yb_lewm_jax_reacher_e100.sh \
  --eval bash /root/data/yyf/ogbench/scripts/eval/20260811_eval_yb_lewm_jax_reacher_e100_cem300x30.sh

bash scripts/recorded_run.sh EXP-013 'LeWMJAX_lance_tworoom_bs128_e100_seed3072_fs5_h3_sigreg009_cem300x30' EXP-013-R04 \
  --train bash /root/data/yyf/ogbench/scripts/train/20260811_train_yb_lewm_jax_tworoom_e100.sh \
  --eval bash /root/data/yyf/ogbench/scripts/eval/20260811_eval_yb_lewm_jax_tworoom_e100_cem300x30.sh
```

JAX planner 的首次调用包含 XLA 编译成本；后续 replan 复用同一可执行图。不要把 MPPI 结果混入本组原版 CEM 基线，MPPI 应使用新的 EXP-ID 和固定 Bash 单独记录。
