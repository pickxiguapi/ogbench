"""Train GCIQL-Chunk with independently shared frozen LeWM representations."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np

from agents.gciql_chunk_lewm import LeWMGCIQLChunkAgent, get_config
from lewm_jax import ARCHITECTURE, LeWM
from utils.datasets import GCChunkDataset
from utils.flax_utils import save_agent
from utils.lewm_dataset import make_lewm_lance_datasets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--lewm_checkpoint', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--share_q_encoder', action='store_true')
    parser.add_argument('--share_v_encoder', action='store_true')
    parser.add_argument('--share_pi_encoder', action='store_true')
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
    parser.add_argument('--actor_loss', choices=('awr', 'ddpgbc'), default='awr')
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--pixel_encoder', default='impala_small')
    parser.add_argument('--p_aug', type=float, default=0.5)
    parser.add_argument('--validation_fraction', type=float, default=0.05)
    return parser.parse_args()


def load_frozen_lewm(checkpoint):
    payload = flax.serialization.msgpack_restore(Path(checkpoint).read_bytes())
    config = payload['config']
    if config.get('architecture') != ARCHITECTURE:
        raise ValueError(
            f'Checkpoint architecture {config.get("architecture")!r} is not '
            f'{ARCHITECTURE!r}.'
        )
    dtype = {
        'bf16': jnp.bfloat16,
        'float32': jnp.float32,
    }[config.get('precision', 'bf16')]
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


def main():
    args = parse_args()
    if not any(
        (args.share_q_encoder, args.share_v_encoder, args.share_pi_encoder)
    ):
        raise ValueError('Enable at least one of Q, V, or pi LeWM sharing.')
    if not 0.0 <= args.p_aug <= 1.0:
        raise ValueError('--p_aug must be in [0, 1].')

    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, lewm_variables, lewm_config = load_frozen_lewm(args.lewm_checkpoint)

    config = get_config()
    config.lr = args.lr
    config.batch_size = args.batch_size
    config.discount = args.discount
    config.expectile = args.expectile
    config.tau = args.tau
    config.chunk_size = args.chunk_size
    config.actor_loss = args.actor_loss
    config.alpha = args.alpha
    config.encoder = args.pixel_encoder
    config.p_aug = 0.0  # Pixel-only augmentation is applied after LeWM encoding.
    config.frame_stack = None
    config.share_q_encoder = args.share_q_encoder
    config.share_v_encoder = args.share_v_encoder
    config.share_pi_encoder = args.share_pi_encoder
    config.latent_dim = int(lewm_config['embed_dim'])

    train_base, val_base = make_lewm_lance_datasets(
        args.dataset_path,
        validation_fraction=args.validation_fraction,
    )
    train_dataset = GCChunkDataset(train_base, config, preprocess_frame_stack=False)
    val_dataset = GCChunkDataset(val_base, config, preprocess_frame_stack=False)

    np.random.seed(args.seed)
    example = train_dataset.sample(1, evaluation=True)
    agent = LeWMGCIQLChunkAgent.create(
        args.seed,
        jnp.asarray(example['observations']),
        jnp.zeros((1, config.latent_dim), dtype=jnp.float32),
        jnp.asarray(example['actions'], dtype=jnp.float32),
        config,
    )

    @jax.jit
    def encode_pixels(pixels):
        return model.apply(
            lewm_variables,
            pixels,
            train=False,
            method=model.encode_pixels,
        ).astype(jnp.float32)

    pixel_keys = (
        'observations',
        'next_observations',
        'value_goals',
        'actor_goals',
    )

    def prepare_batch(sample, training):
        batch_size = sample['observations'].shape[0]
        raw_pixels = [jnp.asarray(sample[key]) for key in pixel_keys]
        latents = encode_pixels(jnp.concatenate(raw_pixels, axis=0))
        latent_parts = jnp.split(
            latents,
            tuple(batch_size * i for i in range(1, len(pixel_keys))),
            axis=0,
        )

        batch = {key: np.array(sample[key], copy=True) for key in pixel_keys}
        if training and args.p_aug > 0 and np.random.rand() < args.p_aug:
            train_dataset.augment(batch, list(pixel_keys))
        batch = {key: jnp.asarray(value) for key, value in batch.items()}
        batch.update(
            {
                f'lewm_{key}': value
                for key, value in zip(pixel_keys, latent_parts)
            }
        )
        for key in ('actions', 'rewards', 'masks'):
            batch[key] = jnp.asarray(sample[key], dtype=jnp.float32)
        return batch

    metadata = {
        'entrypoint': 'train_lewm_gciql_chunk.py',
        'dataset_path': args.dataset_path,
        'lewm_checkpoint': args.lewm_checkpoint,
        'train_steps': args.train_steps,
        'seed': args.seed,
        'agent': dict(config),
        'representation': {
            'module': 'LeWM.encode_pixels',
            'output': 'post_projector',
            'latent_dim': int(lewm_config['embed_dim']),
            'frozen': True,
            'q': 'lewm' if args.share_q_encoder else args.pixel_encoder,
            'v': 'lewm' if args.share_v_encoder else args.pixel_encoder,
            'pi': 'lewm' if args.share_pi_encoder else args.pixel_encoder,
            'downstream_heads_shared': False,
        },
        'augmentation': {
            'pixel_branches_probability': args.p_aug,
            'lewm_branch_probability': 0.0,
        },
    }
    (output_dir / 'flags.json').write_text(json.dumps(metadata, indent=2) + '\n')

    fields = [
        'step',
        'loss',
        'value/value_loss',
        'critic/critic_loss',
        'actor/actor_loss',
        'actor/adv',
        'actor/mse',
        'actor/std',
        'validation/loss',
        'steps_per_second',
        'total_seconds',
    ]
    started = time.time()
    interval_started = started
    with (output_dir / 'train.csv').open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for step in range(1, args.train_steps + 1):
            batch = prepare_batch(train_dataset.sample(args.batch_size), training=True)
            agent, info = agent.update(batch)

            if step == 1 or step % args.log_interval == 0:
                elapsed = time.time() - interval_started
                val_batch = prepare_batch(
                    val_dataset.sample(args.batch_size, evaluation=True),
                    training=False,
                )
                val_loss, _ = agent.total_loss(val_batch, grad_params=None)
                row = {
                    key: float(value)
                    for key, value in jax.device_get(info).items()
                }
                row.update(
                    {
                        'step': step,
                        'loss': sum(
                            row[key]
                            for key in (
                                'value/value_loss',
                                'critic/critic_loss',
                                'actor/actor_loss',
                            )
                        ),
                        'validation/loss': float(jax.device_get(val_loss)),
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
                save_agent(agent, output_dir, step)


if __name__ == '__main__':
    main()
