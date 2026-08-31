# OGBench Local Rules

本文件作用于 `ogbench/`；与上层 `AGENTS.md` 同时生效。

## 最终方法边界

- LeWM世界模型训练入口只有 `impls/train_lewm_jax.py`。
- GCIQL-Chunk 训练入口只有 `impls/train_gciql_chunk.py`。
- Frozen LeWM latent subgoal GCBC 训练入口只有 `impls/train_latent_subgoal_gcbc.py`。
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

### LeWM-4Tasks 默认混合 Checkpoint 映射

后续 LeWM-4Tasks、共享表征和 latent subgoal generator 实验统一使用以下混合 checkpoint：**PushT 使用 training seed 666，Cube、Reacher、TwoRoom 使用 training seed 3072**。除非实验名称和 Bash 顶部明确声明 checkpoint 消融，不得自行替换为其他 seed。

```bash
# PushT
pusht_lewm="$LEWM_SEED666_ROOT/2026-08-19_23_LeWMJAX_impala_lance_pusht_expert_bs128_e10_seed666/weights_epoch_10.msgpack"

# 其他三任务
cube_lewm="$LEWM_SEED3072_ROOT/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"

reacher_lewm="$LEWM_SEED3072_ROOT/LeWMJAX_impala_lance_reacher_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"

tworoom_lewm="$LEWM_SEED3072_ROOT/LeWMJAX_impala_lance_tworoom_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack"
```

各服务器应使用本机镜像根目录，不得通过重命名或覆盖文件把 seed 666 伪装成 seed 3072：

| 服务器 | `LEWM_SEED666_ROOT` | `LEWM_SEED3072_ROOT` |
|---|---|---|
| Server 23 | `/data/dzb/stablewm-data/lewm-jax` | `/data/dzb/stablewm-data/lewm-jax-runs` |
| 英博云 `yingbo1` | `/root/data/yyf/lewm-jax-seed666-s23` | `/root/data/yyf/lewm-jax-seed3072-s23` |
| A800 node2、node3、node4 | `/data-training/yyf/models/lewm-jax-seed666` | `/data-training/yyf/models/lewm-jax-seed3072` |

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

## GCIQL-Chunk DDPG+BC LeWM-4Tasks 多 Seed 结果

Policy training seeds `0/42/777`、evaluation seeds `0/1/42`、每格50 episodes。Policy 为 independent DDPG+BC、chunk5、alpha1、100k；Guided 使用 seed3072 LeWM 与 CEM300x5、H5/RH1、min-over-horizon。

| 模式 | Cube | PushT | Reacher | TwoRoom | 四任务平均 | Training-seed std |
|---|---:|---:|---:|---:|---:|---:|
| Policy-only | 94.44 | 75.56 | 91.56 | 100.00 | 90.39 | 1.08 |
| Guided | 88.44 | 83.78 | 100.00 | 97.33 | 92.39 | 0.35 |

三个 policy seeds 的宏平均为 Policy-only `89.33/90.33/91.50`，Guided `92.50/92.67/92.00`。DDPG+BC 没有出现 AWR 某些 seeds 在 Reacher/PushT 上的灾难性失败。Guidance 对 PushT/Reacher 稳定有利，对 Cube/TwoRoom 稳定不利；完整结果见 `reports/2026-08-29-ddpgbc-multiseed-policy-guided.md`。

## 正式评测入口

- `impls/eval_lewm_4tasks.py`：LeWM-4Tasks 的 policy、LeWM、predicted-subgoal LeWM、oracle-subgoal LeWM、guided、native-Q 六种评测模式。
- `impls/eval_ogbench_env_8tasks.py`：OGBench-Env-8Tasks 的 policy、LeWM、guided、native-Q 四种评测模式。
- `impls/gciql_chunk_policy.py`：两套评测共用的 checkpoint 加载和 policy/native-Q adapter，不是命令入口。
- `impls/lewm_jax/planner.py` 是两套评测共用的 planner 实现，不是正式命令入口。
- shared policy 的 Q 只能通过公开 `score_actions()` 接口调用。`qv`/`all` 的 shared-Q 策略与 planner 必须引用同一个规范化 LeWM checkpoint 路径；

## 实验 Bash

- 所有新训练与评测只能通过 `exp/` 下的 Bash 发起；不得直接运行 Python 命令。
- `exp/train/` 保存正式训练 Bash；`exp/eval/lewm_4tasks/` 与 `exp/eval/ogbench_env_8tasks/` 分别保存两套评测协议；`exp/preprocess/lewm_latents/` 保存 frozen LeWM latent 数据集转换 Bash。
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
| `exp/eval/lewm_4tasks/20260829_eval_node4_ddpgbc_multiseed_policy_guided.sh` | node4 评测 DDPG+BC policy seeds 0/42/777 × eval seeds 0/1/42 的 Policy-only 与 Guided |
| `exp/eval/lewm_4tasks/20260831_eval_yb_lewm_latent_subgoal_moh.sh` | 英博云用 K10 predicted latent subgoal 评测纯 CEM MoH，并与 mixed-checkpoint global-goal CEM 严格配对 |
| `exp/eval/lewm_4tasks/20260831_eval_yb_lewm_oracle_subgoal_moh.sh` | 英博云用 evaluation trajectory 的 GT K10 waypoint 评测纯 CEM MoH，测量 subgoal 可利用性的 oracle 上界 |
| `exp/eval/lewm_4tasks/20260831_eval_yb_lewm_flow_transformer_subgoal_moh.sh` | 英博云用 200k Transformer-CFM K10 predicted latent subgoal 评测纯 CEM MoH，与 MLP/global/oracle 严格配对 |
| `exp/eval/lewm_4tasks/20260831_eval_yb_lewm_flow_transformer_subgoal_terminal.sh` | 英博云用 200k Transformer-CFM K10 predicted latent subgoal 评测 H5 terminal CEM；除不用 MoH 外与 Flow-MoH 严格配对 |
| `exp/eval/lewm_4tasks/20260831_eval_yb_lewm_flow_transformer_subgoal_fixed_k10.sh` | 英博云固定比较第 2 个 rollout checkpoint (t+10) 与 200k Transformer-CFM K10 subgoal；其余协议与 Flow-MoH 严格配对 |
| `exp/eval/lewm_4tasks/20260831_eval_node4_lewm_latent_path_flow_k10_terminal.sh` | node4 只取 200k LatentPathFlow 的 K10 token 替代 K25 global goal，以 H5 terminal cost 四卡评测，不使用 K5 或 MoH |
| `exp/eval/lewm_4tasks/20260831_eval_node4_lewm_latent_path_flow_k10_h2_terminal.sh` | node4 只取 200k LatentPathFlow K10 token，H2 terminal 对齐 t+10，并在每次 RH1 replan（5步）重新预测 K10；不使用 K5 或 MoH |
| `exp/eval/ogbench_env_8tasks/20260823_eval_node2_ogbench_env_8tasks.sh` | OGBench-Env-8Tasks 主评测 |
| `exp/train/20260823_reproduce_yb_lewm_4tasks_main_matrix.sh` | 顺序复现 LeWM-4Tasks 四种训练设计 |
| `exp/train/20260823_reproduce_node2_ogbench_env_8tasks_main_matrix.sh` | 顺序复现 OGBench-Env-8Tasks 四种训练设计 |
| `exp/eval/lewm_4tasks/20260823_reproduce_yb_lewm_4tasks_main_matrix.sh` | 复现 LeWM-4Tasks 主评测矩阵 |
| `exp/eval/ogbench_env_8tasks/20260823_reproduce_node2_ogbench_env_8tasks_main_matrix.sh` | 复现 OGBench-Env-8Tasks 主评测矩阵 |
| `exp/preprocess/lewm_latents/20260830_precompute_yb_pusht_lewm_s666_z192.sh` | 英博云生成 PushT seed666 frozen LeWM z192 数据集 |
| `exp/preprocess/lewm_latents/20260830_precompute_node2_cube_lewm_s3072_z192.sh` | node2 生成 Cube seed3072 frozen LeWM z192 数据集 |
| `exp/preprocess/lewm_latents/20260830_precompute_node3_reacher_lewm_s3072_z192.sh` | node3 生成 Reacher seed3072 frozen LeWM z192 数据集 |
| `exp/preprocess/lewm_latents/20260830_precompute_node4_tworoom_lewm_s3072_z192.sh` | node4 生成 TwoRoom seed3072 frozen LeWM z192 数据集 |
| `exp/train/latent_subgoal/20260830_run_yb_pusht_latent_gcbc_k10_s0.sh` | 英博云生成 PushT cache 并训练 K10 latent subgoal GCBC seed0 |
| `exp/train/latent_subgoal/20260830_run_node2_cube_latent_gcbc_k10_s0.sh` | node2 生成 Cube cache 并训练 K10 latent subgoal GCBC seed0 |
| `exp/train/latent_subgoal/20260830_run_node3_reacher_latent_gcbc_k10_s0.sh` | node3 生成 Reacher cache 并训练 K10 latent subgoal GCBC seed0 |
| `exp/train/latent_subgoal/20260830_run_node4_tworoom_latent_gcbc_k10_s0.sh` | node4 生成 TwoRoom cache 并训练 K10 latent subgoal GCBC seed0 |
| `exp/train/latent_subgoal/20260831_run_yb_pusht_flow_transformer_k10_s0.sh` | 英博云训练 PushT K10 Transformer-CFM subgoal generator seed0 |
| `exp/train/latent_subgoal/20260831_run_node2_cube_flow_transformer_k10_s0.sh` | node2 训练 Cube K10 Transformer-CFM subgoal generator seed0 |
| `exp/train/latent_subgoal/20260831_run_node3_reacher_flow_transformer_k10_s0.sh` | node3 训练 Reacher K10 Transformer-CFM subgoal generator seed0 |
| `exp/train/latent_subgoal/20260831_run_node4_tworoom_flow_transformer_k10_s0.sh` | node4 训练 TwoRoom K10 Transformer-CFM subgoal generator seed0 |
| `exp/train/latent_subgoal/20260831_run_yb_pusht_latent_path_flow_k10_s0.sh` | 英博云训练 PushT K5/K10 LeFlow-style LatentPathFlow seed0 |
| `exp/train/latent_subgoal/20260831_run_node2_cube_latent_path_flow_k10_s0.sh` | node2 训练 Cube K5/K10 LeFlow-style LatentPathFlow seed0 |
| `exp/train/latent_subgoal/20260831_run_node3_reacher_latent_path_flow_k10_s0.sh` | node3 训练 Reacher K5/K10 LeFlow-style LatentPathFlow seed0 |
| `exp/train/latent_subgoal/20260831_run_node4_tworoom_latent_path_flow_k10_s0.sh` | node4 训练 TwoRoom K5/K10 LeFlow-style LatentPathFlow seed0 |
| `exp/train/latent_subgoal/20260831_run_node4_4tasks_latent_path_flow_k10_s0.sh` | node4 GPU0–3 并行训练四任务 K5/K10 LeFlow-style LatentPathFlow；可用 `TASKS` 选择子集 |

## Frozen LeWM latent 数据集

- 正式转换入口只有 `impls/precompute_lewm_latents.py`，正式运行必须使用 `exp/preprocess/lewm_latents/` 下对应服务器和任务的 Bash。
- 每个源数据 row 保存一个 `z = LeWM.encode_pixels(pixels, train=False)`，形状固定为 `[N, 192]`、正式 dtype 固定为 `float32`；不得保存 predictor rollout 结果冒充 encoder target latent。
- 输入图像必须来自训练 LeWM 时使用的 JPEG-backed Lance `pixels`，不能改用原始 HDF5 pixels。输出 HDF5 不复制 pixels，但保留源 HDF5 的非像素字段，并从 Lance 补齐 `episode_idx`、`step_idx`、`ep_offset`、`ep_len`、`source_row`。
- latent cache 与 checkpoint SHA-256 强绑定；PushT 使用 seed666，Cube、Reacher、TwoRoom 使用 seed3072。不同任务即使同为 192 维也不是共享坐标系。
- 英博云输出根目录为 `/root/data/yyf/lewm-latent-datasets/`；A800 node2、node3、node4 输出根目录为 `/data-training/yyf/datasets/lewm-latents/`。
- 正式文件名分别为 `pusht_expert_train__lewm_s666_e10_z192.h5`、`cube_single_expert__lewm_s3072_e10_z192.h5`、`reacher__lewm_s3072_e10_z192.h5`、`tworoom__lewm_s3072_e10_z192.h5`。
- 转换中间文件以 `.incomplete` 结尾并通过 `encoded_rows` 断点续跑；只有所有 z 通过 finite 检查并写入统计量后才原子改名为正式 `.h5`。

## Latent Subgoal GCBC（K=10）

- 定稿设计见 `reports/2026-08-30-latent-subgoal-gcbc-k10-design.md`。
- 四任务分别训练独立 generator；输入为 `[z_t, z_g]`，三层 512 hidden MLP 直接输出完整 192 维 `z_subgoal`，禁止改成 residual output。
- goal sampling 固定为 HIQL 同轨迹未来均匀采样：`p_trajgoal=1.0`、`p_randomgoal=0.0`、`geom_sample=false`；监督 target 为 `z[min(t+10, g)]`。
- 唯一训练 loss 为 raw LeWM latent MSE；cosine similarity 只记录为验证指标。
- seed0 正式设置固定为 100k steps、batch size 1024、AdamW、peak lr 3e-4、warmup 2k、cosine decay 到 3e-5、episode 95/5 split。
- 正式训练必须通过 `exp/train/latent_subgoal/` 下对应服务器的流水线 Bash；流水线先完成或复用 latent cache，再开始训练。

## Latent Subgoal Transformer-CFM（K=10）

- 新版设计见 `reports/2026-08-31-latent-subgoal-flow-transformer-k10-design.md`；旧 MLP checkpoint 仍兼容读取，但新旧 architecture/loss 标记和输出目录必须分开。
- 条件为 `z_t,z_g`，target 和 goal sampling 与 MLP 版完全相同。训练从 `epsilon ~ N(0,I)` 构造 `z_tau=(1-tau)epsilon+tau z_target`，用 MSE 回归速度 `z_target-epsilon`。
- 模型固定为 4 token Transformer Encoder：`z_t,z_g,z_tau,tau`；`d_model=384`、8 layers、8 heads、FFN 1536，参数量约 14.64M，不预测 residual subgoal。
- 推理使用 EMA 参数，从确定性派生的高斯噪声出发做 16-step Heun ODE integration；同 evaluation seed、env index、generation count 必须得到相同 subgoal。
- seed0 正式设置为 200k steps、batch size 1024、AdamW、peak lr 1e-4、warmup 5k、cosine decay 到 1e-5、EMA 0.9999、episode 95/5 split。

## LatentPathFlow（K5/K10）

- 定稿设计见 `reports/2026-08-31-latent-path-flow-k5-k10-design.md`；新模型不得覆盖旧单点 Transformer-CFM checkpoint 或输出目录。
- 条件为 `z_t,z_g`，监督路径固定为 `[z_min(t+5,g), z_min(t+10,g)]`，goal sampling 仍为 HIQL 同轨迹未来均匀采样。
- 唯一 loss 为 conditional flow matching MSE；不训练 inverse dynamics，不添加 LeWM consistency loss。
- 网络固定为 LeFlow-style path-token Transformer：hidden 512、depth 4、8 heads、FFN 2048、time embedding 64、两个 learned waypoint position embeddings。
- 推理使用 EMA 参数和 16-step Euler；seed0 正式设置为 200k、batch size 1024、peak/final lr 1e-4/1e-5、warmup 5k、EMA 0.9999、episode 95/5 split。

## 服务器与 GitHub

- 公共服务器路径仍统一记录在 `scripts/client_env.sh`；实验 Bash 先设置 `CLIENT_ID` 再 source。
- 远程 GitHub `https://github.com/pickxiguapi/ogbench` 的 `main` 是唯一代码基准。
- 服务器无法 pull 时，先在本地对齐远程 `main`，再将本地代码 rsync 到服务器。
- `master` 只用于跟踪上游 `https://github.com/seohongpark/ogbench`，不得提交本地方法代码。
