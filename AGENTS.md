# OGBench Local Rules

本文件作用于 `ogbench/`；与上层 `AGENTS.md` 同时生效。

## 最终方法边界

- LeWM世界模型训练入口只有 `impls/train_lewm_jax.py`。
- GCIQL-Chunk 训练入口只有 `impls/train_gciql_chunk.py`。
- GCIQL-Chunk 表征模式固定为 `independent`、`pi`、`qv`、`all`。
- `pi`、`qv`、`all` 必须加载已经训练好的 LeWM-JAX，并冻结 encoder、projector、predictor 与 batch statistics；只训练策略侧仍可训练的 encoder 和 downstream heads。
- 正式共享消融默认 `p_aug=0.0`。如启用增强，必须先增强同一批像素，再同时送入 pixel encoder 和 frozen LeWM encoder，禁止只增强其中一个分支，注意，如果使用`pi`、`qv`、`all`增强，前提肯定是LeWM本身也被增强训练。
- 执行阶段，也有三种方式：1️⃣ 纯Planning，就是调用LeWM来决策 2️⃣ 纯Policy，就是直接使用Chunk Level的action来决策，跟LeWM没关系 3️⃣ Guidance：GCIQL-Chunk proposal 初始化第一个 action block，LeWM min-over-horizon CEM 完成后续优化。

## LeWM-JAX 4Tasks 三训练 Seed Checkpoint

以下为 LeWM-JAX、IMPALA-Small、10 epoch、batch size 128、frameskip 5、history 3、SigReg 0.09 的主表 checkpoint。表中 SR 是当前实验主表记录值；`Seed 0/42` 位于英博云，`Seed 3072` 的原始训练产物位于 Server 23。

| Training Seed | Task | 主表 SR | 服务器 | Checkpoint |
|---:|---|---:|---|---|
| 0 | PushT | 88 | 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-runs/2026-08-21_yb_LeWMJAX_pusht_impalasmall_lance_bs128_e10_s0_fs5_h3_sigreg009/weights_epoch_10.msgpack` |
| 0 | OGB Cube | 60 | 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-runs/2026-08-21_yb_LeWMJAX_cube_impalasmall_lance_bs128_e10_s0_fs5_h3_sigreg009/weights_epoch_10.msgpack` |
| 0 | Reacher | 80 | 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-runs/2026-08-21_yb_LeWMJAX_reacher_impalasmall_lance_bs128_e10_s0_fs5_h3_sigreg009/weights_epoch_10.msgpack` |
| 0 | TwoRoom | 90 | 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-runs/2026-08-21_yb_LeWMJAX_tworoom_impalasmall_lance_bs128_e10_s0_fs5_h3_sigreg009/weights_epoch_10.msgpack` |
| 42 | PushT | 90 | 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-runs/2026-08-21_yb_LeWMJAX_pusht_impalasmall_lance_bs128_e10_s42_fs5_h3_sigreg009/weights_epoch_10.msgpack` |
| 42 | OGB Cube | 68 | 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-runs/2026-08-21_yb_LeWMJAX_cube_impalasmall_lance_bs128_e10_s42_fs5_h3_sigreg009/weights_epoch_10.msgpack` |
| 42 | Reacher | 80 | 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-runs/2026-08-21_yb_LeWMJAX_reacher_impalasmall_lance_bs128_e10_s42_fs5_h3_sigreg009/weights_epoch_10.msgpack` |
| 42 | TwoRoom | 88 | 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-runs/2026-08-21_yb_LeWMJAX_tworoom_impalasmall_lance_bs128_e10_s42_fs5_h3_sigreg009/weights_epoch_10.msgpack` |
| 3072 | PushT | 90 | Server 23 | `/data/dzb/stablewm-data/lewm-jax-runs/LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack` |
| 3072 | OGB Cube | 70 | Server 23 | `/data/dzb/stablewm-data/lewm-jax-runs/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack` |
| 3072 | Reacher | 86 | Server 23 | `/data/dzb/stablewm-data/lewm-jax-runs/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack` |
| 3072 | TwoRoom | 86 | Server 23 | `/data/dzb/stablewm-data/lewm-jax-runs/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack` |

Seed 3072 四个 checkpoint 有以下镜像；已对英博云、Server 23、A800 node1 和 node2 的文件做 SHA-256 校验，逐任务一致：

| 服务器 | 镜像根目录 |
|---|---|
| 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-seed3072-s23/` |
| A800 node1 | `/data-training/yyf/lewm-jax-seed3072/` |
| A800 node2、node3、node4 | `/data-training/yyf/models/lewm-jax-seed3072/` |

注意：Seed 3072 的主表 SR 与 Server 23 的原始 `eval_cem300x30_seed42/` JSON 精确对应。Seed 0/42 的 PushT、Reacher、TwoRoom 可与英博云相邻评测 JSON 对应，但两个 Cube checkpoint 当前相邻的 `eval_cem300x30_seed42_20260822/cube.json` 分别记录 40 和 42，而非主表的 60 和 68；使用 Cube 主表数字前必须先核对评测轮次，不能把 training seed 与其他评测结果混用。

补充：除了3072的Cube分数是对的，别的都不太对，因为数据集broken了

## GCIQL-Chunk-AWR OGBench-Env-8Tasks 三训练 Seed Checkpoint

这组主结果是 OGBench 原生视觉环境的 8 Tasks，不是 LeWM-4Tasks。训练设置为 GCIQL-Chunk-AWR、IMPALA-Small、chunk size 5、batch size 512、alpha 3、expectile 0.9、`p_aug=0.5`、总训练量 500K。三个 training seed 为 0、42、999；每个任务都保留 300K、400K、500K checkpoint，共 `8 × 3 × 3 = 72` 个文件。实验表中每个 seed 的 `Mean` 是 300K/400K/500K 三次评测的均值，不存在独立的 Mean checkpoint。

下面路径中的 `params_{300000,400000,500000}.pkl` 表示同一目录下已逐个核验存在的三个文件。

### Training seed 0：英博云 `yingbo1`

- Cube Single Play：`/root/data/yyf/ogbench-gciql-chunk-awr-runs/2026-08-19_yb_GCAWR_cs_play_k5_bs512_s500k_s0_a3_e09_aug05/OGBench/2026-08-19_yb_GCAWR_cs_play_k5_bs512_s500k_s0_a3_e09_aug05/sd000_20260819_080318/params_{300000,400000,500000}.pkl`
- Cube Double Play：`/root/data/yyf/ogbench-gciql-chunk-awr-runs/2026-08-19_yb_GCAWR_cd_play_k5_bs512_s500k_s0_a3_e09_aug05/OGBench/2026-08-19_yb_GCAWR_cd_play_k5_bs512_s500k_s0_a3_e09_aug05/sd000_20260819_080318/params_{300000,400000,500000}.pkl`
- Cube Triple Play：`/root/data/yyf/ogbench-gciql-chunk-awr-runs/2026-08-19_yb_GCAWR_ct_play_k5_bs512_s500k_s0_a3_e09_aug05/OGBench/2026-08-19_yb_GCAWR_ct_play_k5_bs512_s500k_s0_a3_e09_aug05/sd000_20260819_080318/params_{300000,400000,500000}.pkl`
- Scene Play：`/root/data/yyf/ogbench-gciql-chunk-awr-runs/2026-08-19_yb_GCAWR_scene_play_k5_bs512_s500k_s0_a3_e09_aug05/OGBench/2026-08-19_yb_GCAWR_scene_play_k5_bs512_s500k_s0_a3_e09_aug05/sd000_20260819_080318/params_{300000,400000,500000}.pkl`
- Cube Single Noisy：`/root/data/yyf/ogbench-gciql-chunk-awr-runs/2026-08-19_yb_GCAWR_cs_noisy_k5_bs512_s500k_s0_a3_e09_aug05/OGBench/2026-08-19_yb_GCAWR_cs_noisy_k5_bs512_s500k_s0_a3_e09_aug05/sd000_20260819_080318/params_{300000,400000,500000}.pkl`
- Cube Double Noisy：`/root/data/yyf/ogbench-gciql-chunk-awr-runs/2026-08-19_yb_GCAWR_cd_noisy_k5_bs512_s500k_s0_a3_e09_aug05/OGBench/2026-08-19_yb_GCAWR_cd_noisy_k5_bs512_s500k_s0_a3_e09_aug05/sd000_20260819_080318/params_{300000,400000,500000}.pkl`
- Cube Triple Noisy：`/root/data/yyf/ogbench-gciql-chunk-awr-runs/2026-08-19_yb_GCAWR_ct_noisy_k5_bs512_s500k_s0_a3_e09_aug05/OGBench/2026-08-19_yb_GCAWR_ct_noisy_k5_bs512_s500k_s0_a3_e09_aug05/sd000_20260819_080318/params_{300000,400000,500000}.pkl`
- Scene Noisy：`/root/data/yyf/ogbench-gciql-chunk-awr-runs/2026-08-19_yb_GCAWR_scene_noisy_k5_bs512_s500k_s0_a3_e09_aug05/OGBench/2026-08-19_yb_GCAWR_scene_noisy_k5_bs512_s500k_s0_a3_e09_aug05/sd000_20260819_080318/params_{300000,400000,500000}.pkl`

### Training seed 42：A800 `node2`

- Cube Single Play：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-21_node2_GCAWR_cs_play_k5_bs512_s500k_s42_a3_e09_aug05/OGBench/2026-08-21_node2_GCAWR_cs_play_k5_bs512_s500k_s42_a3_e09_aug05/sd042_20260821_135042/params_{300000,400000,500000}.pkl`
- Cube Double Play：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-21_node2_GCAWR_cd_play_k5_bs512_s500k_s42_a3_e09_aug05/OGBench/2026-08-21_node2_GCAWR_cd_play_k5_bs512_s500k_s42_a3_e09_aug05/sd042_20260821_182708/params_{300000,400000,500000}.pkl`
- Cube Triple Play：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-21_node2_GCAWR_ct_play_k5_bs512_s500k_s42_a3_e09_aug05/OGBench/2026-08-21_node2_GCAWR_ct_play_k5_bs512_s500k_s42_a3_e09_aug05/sd042_20260821_182708/params_{300000,400000,500000}.pkl`
- Scene Play：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-21_node2_GCAWR_scene_play_k5_bs512_s500k_s42_a3_e09_aug05/OGBench/GCAWR_scene_play_s42/sd042_20260821_183034/params_{300000,400000,500000}.pkl`
- Cube Single Noisy：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-21_node2_GCAWR_cs_noisy_k5_bs512_s500k_s42_a3_e09_aug05/OGBench/2026-08-21_node2_GCAWR_cs_noisy_k5_bs512_s500k_s42_a3_e09_aug05/sd042_20260821_182708/params_{300000,400000,500000}.pkl`
- Cube Double Noisy：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-21_node2_GCAWR_cd_noisy_k5_bs512_s500k_s42_a3_e09_aug05/OGBench/2026-08-21_node2_GCAWR_cd_noisy_k5_bs512_s500k_s42_a3_e09_aug05/sd042_20260821_182708/params_{300000,400000,500000}.pkl`
- Cube Triple Noisy：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-21_node2_GCAWR_ct_noisy_k5_bs512_s500k_s42_a3_e09_aug05/OGBench/2026-08-21_node2_GCAWR_ct_noisy_k5_bs512_s500k_s42_a3_e09_aug05/sd042_20260821_182708/params_{300000,400000,500000}.pkl`
- Scene Noisy：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-21_node2_GCAWR_scene_noisy_k5_bs512_s500k_s42_a3_e09_aug05/OGBench/GCAWR_scene_noisy_s42/sd042_20260821_183034/params_{300000,400000,500000}.pkl`

### Training seed 999：A800 `node3`

- Cube Single Play：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-21_node3_GCAWR_cs_play_k5_bs512_s500k_s999_a3_e09_aug05/OGBench/2026-08-21_node3_GCAWR_cs_play_k5_bs512_s500k_s999_a3_e09_aug05/sd999_20260821_135041/params_{300000,400000,500000}.pkl`
- Cube Double Play：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-21_node3_GCAWR_cd_play_k5_bs512_s500k_s999_a3_e09_aug05/OGBench/GCAWR_cd_play_s999/sd999_20260822_023505/params_{300000,400000,500000}.pkl`
- Cube Triple Play：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-21_node3_GCAWR_ct_play_k5_bs512_s500k_s999_a3_e09_aug05/OGBench/GCAWR_ct_play_s999/sd999_20260822_023505/params_{300000,400000,500000}.pkl`
- Scene Play：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-21_node3_GCAWR_scene_play_k5_bs512_s500k_s999_a3_e09_aug05/OGBench/GCAWR_scene_play_s999/sd999_20260822_023505/params_{300000,400000,500000}.pkl`
- Cube Single Noisy：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-21_node3_GCAWR_cs_noisy_k5_bs512_s500k_s999_a3_e09_aug05/OGBench/GCAWR_cs_noisy_s999/sd999_20260822_023505/params_{300000,400000,500000}.pkl`
- Cube Double Noisy：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-21_node3_GCAWR_cd_noisy_k5_bs512_s500k_s999_a3_e09_aug05/OGBench/GCAWR_cd_noisy_s999/sd999_20260821_183125/params_{300000,400000,500000}.pkl`
- Cube Triple Noisy：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-21_node3_GCAWR_ct_noisy_k5_bs512_s500k_s999_a3_e09_aug05/OGBench/GCAWR_ct_noisy_s999/sd999_20260821_183757/params_{300000,400000,500000}.pkl`
- Scene Noisy：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-21_node3_GCAWR_scene_noisy_k5_bs512_s500k_s999_a3_e09_aug05/OGBench/GCAWR_scene_noisy_s999/sd999_20260821_183226/params_{300000,400000,500000}.pkl`

### `sdepstd` 消融：training seed 0，A800 `node2`

这套使用 state-dependent standard deviation（`sdepstd`），共有 8 Tasks × 300K/400K/500K = 24 个 checkpoint。它必须标注为 `sdepstd`，不属于上述 seed 0/42/999 三 seed 主表，也不得代替英博云的标准 seed 0。

- Cube Single Play：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-22_node2_GCAWR_cs_play_k5_bs512_s500k_s0_a3_e09_aug05_sdepstd/OGBench/node2_GCAWR_s0_sdstd/sd000_20260822_130149/params_{300000,400000,500000}.pkl`
- Cube Double Play：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-22_node2_GCAWR_cd_play_k5_bs512_s500k_s0_a3_e09_aug05_sdepstd/OGBench/node2_GCAWR_s0_sdstd/sd000_20260822_130149/params_{300000,400000,500000}.pkl`
- Cube Triple Play：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-22_node2_GCAWR_ct_play_k5_bs512_s500k_s0_a3_e09_aug05_sdepstd/OGBench/node2_GCAWR_s0_sdstd/sd000_20260822_130149/params_{300000,400000,500000}.pkl`
- Scene Play：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-22_node2_GCAWR_scene_play_k5_bs512_s500k_s0_a3_e09_aug05_sdepstd/OGBench/node2_GCAWR_s0_sdstd/sd000_20260822_130149/params_{300000,400000,500000}.pkl`
- Cube Single Noisy：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-22_node2_GCAWR_cs_noisy_k5_bs512_s500k_s0_a3_e09_aug05_sdepstd/OGBench/node2_GCAWR_s0_sdstd/sd000_20260822_130149/params_{300000,400000,500000}.pkl`
- Cube Double Noisy：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-22_node2_GCAWR_cd_noisy_k5_bs512_s500k_s0_a3_e09_aug05_sdepstd/OGBench/node2_GCAWR_s0_sdstd/sd000_20260822_130149/params_{300000,400000,500000}.pkl`
- Cube Triple Noisy：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-22_node2_GCAWR_ct_noisy_k5_bs512_s500k_s0_a3_e09_aug05_sdepstd/OGBench/node2_GCAWR_s0_sdstd/sd000_20260822_130149/params_{300000,400000,500000}.pkl`
- Scene Noisy：`/data-training/yyf/ogbench-visual-policy-runs/2026-08-22_node2_GCAWR_scene_noisy_k5_bs512_s500k_s0_a3_e09_aug05_sdepstd/OGBench/node2_GCAWR_s0_sdstd/sd000_20260822_130148/params_{300000,400000,500000}.pkl`

### 服务器归属与排除项

- 英博云 `yingbo1`：training seed 0，24/24 个文件齐全。
- A800 `node2`：training seed 42，24/24 个文件齐全。
- A800 `node3`：training seed 999，24/24 个文件齐全。
- A800 `node1`、A800 `node4`、Server 11、Server 23、Server 7002 没有这三套原始 checkpoint。A800 各 node 的 `/data-training` 不应视为共享的同一份目录。
- `node2` 另有上表已记录的 training seed 0 `sdepstd` 消融 checkpoint；它与三 seed 主结果配置不同，不得混入主表或当作 seed 0 镜像。

## GCIQL-Chunk-AWR LeWM-4Tasks independent seed 131/132 关键结果

这组实验是 LeWM-4Tasks 上的纯 independent GCIQL-Chunk-AWR，不加载 LeWM 表征。训练设置为 IMPALA-Small、100K steps、batch size 256、chunk size 5、AWR alpha 3、expectile 0.9、`p_aug=0.5`。英博云 `yingbo1` 上 seed 131 使用 GPU 0–3，seed 132 使用 GPU 4–7，八个任务 checkpoint 均完整保存到 100K：

`/root/data/yyf/lewm-final/gciql-chunk-4tasks/gc4_{cube,pusht,reacher,tworoom}_ind_n100000_b256_a0.5_sd{131,132}/params_100000.pkl`

正式评测使用 LeWM training seed 3072、epoch 10；每项 50 episodes、evaluation seed 42。CEM 设置为 horizon 5、receding horizon 1、300 samples、5 iterations、top-k 30、`min_over_horizon`。结果根目录：

`/root/data/yyf/lewm-final/evals/lewm-4tasks/20260824_gciql_chunk_independent_seeds131_132/`

| 方法 | Policy training seed | PushT | OGB Cube | Reacher | TwoRoom | 四任务平均 |
|---|---:|---:|---:|---:|---:|---:|
| CEM-only | 不适用 | 84 | 76 | 100 | 94 | 88.5 |
| Policy-only | 131 | 74 | 96 | 76 | 100 | 86.5 |
| Policy-only | 132 | 74 | 94 | 86 | 100 | 88.5 |
| CEM + policy | 131 | 88 | 88 | 100 | 100 | 94.0 |
| CEM + policy | 132 | 88 | 94 | 100 | 98 | 95.0 |
| Policy-only 两 seed 均值 | — | 74 | 95 | 81 | 100 | 87.5 |
| CEM + policy 两 seed 均值 | — | 88 | 91 | 100 | 99 | 94.5 |

关键结论：当前协议下 CEM + policy 明显优于单独 CEM 和单独 policy；相对各自 policy-only，四任务平均分别提高 7.5 和 6.5 分。Guidance 只用确定性 policy mode 初始化第一个 action block，后续仍由 LeWM-CEM 优化。

不要把这张表误解为“不同 seed 完全没有波动”：

- CEM-only 不加载 policy checkpoint，因此 policy seed 131/132 对它没有定义；表中的 CEM-only 只需评测一次。
- Policy-only 的 Reacher 在两个 training seed 间相差 10 分，CEM + policy 的 Cube 相差 6 分；只是 Reacher、TwoRoom 等任务接近 100% 的天花板，四任务平均的差异被压缩到 2 分和 1 分。
- 两次评测固定使用同一个 evaluation seed 42。`sample_starts()` 因而选择完全相同的 50 个 dataset episode/start，环境从相同 dataset state 和逐行 seed 恢复；policy-only 使用 `temperature=0.0`，guided proposal 也使用确定性 mode。
- Planner 使用 `paired_plan_keys=True`，CEM 随机 key 由 evaluation seed、environment index 和 plan count 确定。因此不同 checkpoint 共享相同起点和 CEM 随机数，这是刻意降低比较噪声的 paired evaluation；同 checkpoint、同 evaluation seed 的重复运行应当精确或近似复现。
- 50 episodes 的 success rate 粒度是 2 个百分点，且目前只有两个 policy training seed。这足以说明 CEM + policy 在这两个 seed 上稳定领先，但不足以声称总体方差为零；正式报告 training-seed mean ± standard deviation 至少还应纳入配置相同的第三个 policy training seed。若要估计 evaluation 随机性，应另外改变 evaluation seed，不能把同一 evaluation seed 的复跑当作独立样本。

## 正式评测入口

- `impls/eval_lewm_4tasks.py`：LeWM-4Tasks 的 policy、LeWM、guided、native-Q 四种评测模式。
- `impls/eval_ogbench_env_8tasks.py`：OGBench-Env-8Tasks 的 policy、LeWM、guided、native-Q 四种评测模式。
- `impls/gciql_chunk_policy.py`：两套评测共用的 checkpoint 加载和 policy/native-Q adapter，不是命令入口。
- `impls/lewm_jax/planner.py` 是两套评测共用的 planner 实现，不是正式命令入口。
- shared policy 的 Q 只能通过公开 `score_actions()` 接口调用。`qv`/`all` 的 shared-Q 策略与 planner 必须引用同一个规范化 LeWM checkpoint 路径；

## 实验 Bash

- 所有新训练与评测只能通过 `exp/` 下的 Bash 发起；不得直接运行 Python 命令。
- `exp/train/` 保存正式训练 Bash；`exp/eval/lewm_4tasks/` 与 `exp/eval/ogbench_env_8tasks/` 分别保存两套评测协议。
- Bash 顶部必须用中文说明服务器、任务范围、算法、训练量和特殊设置。
- 可调参数集中放在 Bash 开头；真实 Python 命令及关键 flags 必须完整展开。
- 修改 Bash 后至少运行 `bash -n`。
- 历史实验只能放在仓库一级 `backup/`，不得从 backup 发起新实验。
- 在bash里面不要设置冗余的判断

## 当前正式 Bash

| Bash | 用途 |
|---|---|
| `exp/train/20260823_train_yb_lewm_4tasks.sh` | 英博云训练 LeWM-4Tasks canonical LeWM-JAX |
| `exp/train/20260823_train_s23_lewm_cube_pusht_seeds42_777.sh` | Server23 GPU2–5 训练 Cube/PushT × seed42/777 LeWM-JAX |
| `exp/train/20260823_train_yb_gciql_chunk_4tasks.sh` | 英博云训练 LeWM-4Tasks 四种 GCIQL-Chunk 表征模式 |
| `exp/train/20260823_train_node2_lewm_ogbench_env_8tasks.sh` | node2 训练 OGBench-Env-8Tasks canonical LeWM-JAX |
| `exp/train/20260823_train_node2_gciql_chunk_ogbench_env_8tasks.sh` | node2 训练 OGBench-Env-8Tasks 四种 GCIQL-Chunk 表征模式 |
| `exp/eval/lewm_4tasks/20260823_eval_yb_lewm_4tasks.sh` | LeWM-4Tasks 主评测 |
| `exp/eval/lewm_4tasks/20260823_eval_s23_lewm_seed666.sh` | Server23 四卡评测 seed 666 的 LeWM-4Tasks checkpoint |
| `exp/eval/ogbench_env_8tasks/20260823_eval_node2_ogbench_env_8tasks.sh` | OGBench-Env-8Tasks 主评测 |
| `exp/train/20260823_reproduce_yb_lewm_4tasks_main_matrix.sh` | 顺序复现 LeWM-4Tasks 四种训练设计 |
| `exp/train/20260823_reproduce_node2_ogbench_env_8tasks_main_matrix.sh` | 顺序复现 OGBench-Env-8Tasks 四种训练设计 |
| `exp/eval/lewm_4tasks/20260823_reproduce_yb_lewm_4tasks_main_matrix.sh` | 复现 LeWM-4Tasks 主评测矩阵 |
| `exp/eval/ogbench_env_8tasks/20260823_reproduce_node2_ogbench_env_8tasks_main_matrix.sh` | 复现 OGBench-Env-8Tasks 主评测矩阵 |

## 服务器与 GitHub

- 公共服务器路径仍统一记录在 `scripts/client_env.sh`；实验 Bash 先设置 `CLIENT_ID` 再 source。
- 远程 GitHub `https://github.com/pickxiguapi/ogbench` 的 `main` 是唯一代码基准。
- 服务器无法 pull 时，先在本地对齐远程 `main`，再将本地代码 rsync 到服务器。
- `master` 只用于跟踪上游 `https://github.com/seohongpark/ogbench`，不得提交本地方法代码。
