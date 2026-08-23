"""Train GCIQL-Chunk Q/V heads on a frozen pretrained LeWM representation."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from agents.gciql_chunk_lewm_shared import (
    LeWMSharedGCIQLChunkEvaluator,
    get_config,
)
from lewm_jax import ARCHITECTURE, LeWM
from utils.datasets import GCChunkDataset
from utils.flax_utils import save_agent
from utils.lewm_dataset import make_lewm_lance_datasets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--lewm_checkpoint', required=True)
    parser.add_argument('--actor_checkpoint_dir', required=True)
    parser.add_argument('--actor_checkpoint_step', type=int, default=100_000)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--train_steps', type=int, default=100_000)
    parser.add_argument('--save_interval', type=int, default=100_000)
    parser.add_argument('--log_interval', type=int, default=5_000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--expectile', type=float, default=0.9)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--chunk_size', type=int, default=5)
    parser.add_argument('--validation_fraction', type=float, default=0.05)
    return parser.parse_args()


def load_frozen_lewm(checkpoint):
    payload = flax.serialization.msgpack_restore(Path(checkpoint).read_bytes())
    config = payload['config']
    if config.get('architecture') != ARCHITECTURE:
        raise ValueError(
            f'Checkpoint architecture {config.get("architecture")!r} is not {ARCHITECTURE!r}.'
        )
    precision = config.get('precision', 'bf16')
    dtype = {'bf16': jnp.bfloat16, 'float32': jnp.float32}[precision]
    model = LeWM(
        image_size=int(config['image_size']),
        embed_dim=int(config['embed_dim']),
        history_size=int(config['history_size']),
        projector_hidden_dim=int(config.get('projector_hidden_dim', 2048)),
        action_smoothed_dim=int(config.get('action_smoothed_dim', 10)),
        action_mlp_scale=int(config.get('action_mlp_scale', 4)),
        predictor_depth=int(config.get('predictor_depth', 6)),
        predictor_heads=int(config.get('predictor_heads', 16)),
        predictor_dim_head=int(config.get('predictor_dim_head', 64)),
        predictor_mlp_dim=int(config.get('predictor_mlp_dim', 2048)),
        predictor_dropout=float(config.get('predictor_dropout', 0.1)),
        predictor_emb_dropout=float(config.get('predictor_emb_dropout', 0.0)),
        dtype=dtype,
    )
    variables = {'params': payload['params'], 'batch_stats': payload['batch_stats']}
    return model, variables, config


def load_actor_dataset_config(checkpoint_dir, checkpoint_step, chunk_size):
    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / f'params_{checkpoint_step}.pkl').is_file():
        raise FileNotFoundError('GCIQL-Chunk actor checkpoint is missing.')
    flags_path = checkpoint_dir / 'flags.json'
    if not flags_path.is_file():
        raise FileNotFoundError('GCIQL-Chunk actor flags.json is missing.')
    saved = json.loads(flags_path.read_text()).get('agent', {})
    if saved.get('agent_name') != 'gciql_chunk':
        raise ValueError(f'Expected gciql_chunk actor, got {saved.get("agent_name")!r}.')
    if int(saved.get('chunk_size', -1)) != chunk_size:
        raise ValueError('Actor chunk size does not match shared evaluator chunk size.')
    # Preserve the actor checkpoint's goal relabeling distribution and reward
    # convention.  Pixel augmentation is disabled because the frozen LeWM was
    # trained and evaluated in its unaugmented latent coordinate system.
    saved['p_aug'] = 0.0
    return ml_collections.ConfigDict(saved)


def main():
    args = parse_args()
    if args.train_steps <= 0 or args.save_interval <= 0 or args.log_interval <= 0:
        raise ValueError('Training and logging intervals must be positive.')

    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, lewm_variables, lewm_config = load_frozen_lewm(args.lewm_checkpoint)
    dataset_config = load_actor_dataset_config(
        args.actor_checkpoint_dir,
        args.actor_checkpoint_step,
        args.chunk_size,
    )
    train_base, val_base = make_lewm_lance_datasets(
        args.dataset_path,
        validation_fraction=args.validation_fraction,
    )
    train_dataset = GCChunkDataset(
        train_base, dataset_config, preprocess_frame_stack=False
    )
    val_dataset = GCChunkDataset(
        val_base, dataset_config, preprocess_frame_stack=False
    )

    evaluator_config = get_config()
    evaluator_config.lr = args.lr
    evaluator_config.discount = args.discount
    evaluator_config.expectile = args.expectile
    evaluator_config.tau = args.tau
    evaluator_config.chunk_size = args.chunk_size
    evaluator_config.latent_dim = int(lewm_config['embed_dim'])

    np.random.seed(args.seed)
    example = train_dataset.sample(1, evaluation=True)
    evaluator = LeWMSharedGCIQLChunkEvaluator.create(
        args.seed,
        jnp.zeros((1, evaluator_config.latent_dim), dtype=jnp.float32),
        jnp.asarray(example['actions']),
        evaluator_config,
    )

    @jax.jit
    def encode_pixels(pixels):
        return model.apply(
            lewm_variables,
            pixels,
            train=False,
            method=model.encode_pixels,
        ).astype(jnp.float32)

    def encode_batch(batch):
        batch_size = batch['observations'].shape[0]
        pixels = jnp.concatenate(
            [
                jnp.asarray(batch['observations']),
                jnp.asarray(batch['next_observations']),
                jnp.asarray(batch['value_goals']),
            ],
            axis=0,
        )
        latents = encode_pixels(pixels)
        observations, next_observations, goals = jnp.split(
            latents, (batch_size, 2 * batch_size), axis=0
        )
        return {
            'observations': observations,
            'next_observations': next_observations,
            'goals': goals,
            'actions': jnp.asarray(batch['actions'], dtype=jnp.float32),
            'rewards': jnp.asarray(batch['rewards'], dtype=jnp.float32),
            'masks': jnp.asarray(batch['masks'], dtype=jnp.float32),
        }

    metadata = {
        'entrypoint': 'train_lewm_with_gciql_chunk.py',
        'dataset_path': args.dataset_path,
        'train_steps': args.train_steps,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'agent': dict(evaluator_config),
        'shared_encoder': {
            'checkpoint': args.lewm_checkpoint,
            'module': 'LeWM.encode_pixels',
            'output': 'post_projector',
            'latent_dim': int(lewm_config['embed_dim']),
            'frozen': True,
            'batch_stats_frozen': True,
        },
        'actor': {
            'checkpoint_dir': args.actor_checkpoint_dir,
            'checkpoint_step': args.actor_checkpoint_step,
            'encoder_shared_with_lewm': False,
            'frozen': True,
        },
        'augmentation': {'shared_qv_pixels': False},
    }
    (output_dir / 'flags.json').write_text(json.dumps(metadata, indent=2) + '\n')

    csv_path = output_dir / 'train.csv'
    fields = [
        'step',
        'loss',
        'value/value_loss',
        'value/value_mean',
        'value/adv_mean',
        'critic/critic_loss',
        'critic/q1_mean',
        'critic/q2_mean',
        'critic/target_mean',
        'validation/loss',
        'steps_per_second',
        'total_seconds',
    ]
    started = time.time()
    interval_started = started
    with csv_path.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for step in range(1, args.train_steps + 1):
            batch = encode_batch(train_dataset.sample(args.batch_size))
            evaluator, info = evaluator.update(batch)

            if step % args.log_interval == 0 or step == 1:
                elapsed = time.time() - interval_started
                val_batch = encode_batch(
                    val_dataset.sample(args.batch_size, evaluation=True)
                )
                _, val_info = evaluator.total_loss(val_batch, grad_params=None)
                row = {
                    key: float(value)
                    for key, value in jax.device_get(info).items()
                }
                row.update(
                    {
                        'step': step,
                        'validation/loss': float(jax.device_get(val_info['loss'])),
                        'steps_per_second': (
                            (1 if step == 1 else args.log_interval) / elapsed
                        ),
                        'total_seconds': time.time() - started,
                    }
                )
                writer.writerow(row)
                csv_file.flush()
                print(json.dumps(row), flush=True)
                interval_started = time.time()

            if step % args.save_interval == 0 or step == args.train_steps:
                save_agent(evaluator, output_dir, step)

if __name__ == '__main__':
    main()
