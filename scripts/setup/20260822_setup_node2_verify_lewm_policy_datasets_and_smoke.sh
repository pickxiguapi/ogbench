#!/usr/bin/env bash
set -euo pipefail

# A800 node2：等待四份 LeWM policy Lance 数据完整传入，随后各做一次 π-only/shared-all 真实 1-step CPU smoke，通过后释放八任务训练队列。
CLIENT_ID=node2
source /data-training/yyf/ogbench/scripts/client_env.sh
DATA_ROOT="$CLIENT_ROOT/datasets/latent-geometry"
CODE_ROOT="$CLIENT_ROOT/ogbench-visual-policy-runs/code/ogbench-shared-policy"

names=(cube_single_expert.lance pusht_expert_train.lance reacher.lance tworoom.lance)
sizes=(18971745447 14177663968 17202901640 4063804288)
for i in "${!names[@]}"; do
  while [[ ! -e "$DATA_ROOT/${names[$i]}" ]] || [[ $(find "$DATA_ROOT/${names[$i]}" -type f -printf '%s\n' | awk '{s += $1} END {printf "%.0f", s}') -ne ${sizes[$i]} ]]; do sleep 30; done
done

cd "$CODE_ROOT/impls"
for sharing in pi_only shared_all; do
  flags=(--share_pi_encoder)
  [[ "$sharing" == shared_all ]] && flags=(--share_q_encoder --share_v_encoder --share_pi_encoder)
  JAX_PLATFORMS=cpu PYTHONPATH="$CODE_ROOT:$CODE_ROOT/impls" "$PYTHON_BIN" train_lewm_gciql_chunk.py \
    --dataset_path="$DATA_ROOT/cube_single_expert.lance" \
    --lewm_checkpoint="$CLIENT_ROOT/models/lewm-jax-seed3072/LeWMJAX_impala_lance_cube_single_bs128_e10_seed3072_fs5_h3_sigreg009_jpeg95/weights_epoch_10.msgpack" \
    --save_dir="$CLIENT_ROOT/ogbench-visual-policy-runs/code/smoke_${sharing}" "${flags[@]}" \
    --train_steps=1 --save_interval=1 --log_interval=1 --batch_size=1 --p_aug=0
done

touch "$DATA_ROOT/.four_lewm_policy_datasets_ready"
