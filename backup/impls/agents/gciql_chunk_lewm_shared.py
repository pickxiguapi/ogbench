"""GCIQL-Chunk Q/V heads trained in a frozen LeWM latent space."""

from __future__ import annotations

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCValue


class LeWMSharedGCIQLChunkEvaluator(flax.struct.PyTreeNode):
    """Twin-Q and V heads over frozen post-projector LeWM embeddings."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(diff, expectile):
        weight = jnp.where(diff >= 0, expectile, 1.0 - expectile)
        return weight * diff**2

    def value_loss(self, batch, grad_params):
        q1, q2 = self.network.select('target_critic')(
            batch['observations'], batch['goals'], batch['actions']
        )
        q = jnp.minimum(q1, q2)
        value = self.network.select('value')(
            batch['observations'], batch['goals'], params=grad_params
        )
        loss = self.expectile_loss(q - value, self.config['expectile']).mean()
        return loss, {
            'value_loss': loss,
            'value_mean': value.mean(),
            'adv_mean': (q - value).mean(),
        }

    def critic_loss(self, batch, grad_params):
        next_value = self.network.select('value')(
            batch['next_observations'], batch['goals']
        )
        discount = self.config['discount'] ** self.config['chunk_size']
        target = batch['rewards'] + discount * batch['masks'] * next_value
        q1, q2 = self.network.select('critic')(
            batch['observations'],
            batch['goals'],
            batch['actions'],
            params=grad_params,
        )
        loss = ((q1 - target) ** 2 + (q2 - target) ** 2).mean()
        return loss, {
            'critic_loss': loss,
            'q1_mean': q1.mean(),
            'q2_mean': q2.mean(),
            'target_mean': target.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params):
        value_loss, value_info = self.value_loss(batch, grad_params)
        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        info = {f'value/{key}': value for key, value in value_info.items()}
        info.update({f'critic/{key}': value for key, value in critic_info.items()})
        info['loss'] = value_loss + critic_loss
        return value_loss + critic_loss, info

    @jax.jit
    def update(self, batch):
        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params)

        network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        target_params = jax.tree_util.tree_map(
            lambda params, target: (
                self.config['tau'] * params + (1.0 - self.config['tau']) * target
            ),
            network.params['modules_critic'],
            network.params['modules_target_critic'],
        )
        network.params['modules_target_critic'] = target_params
        return self.replace(network=network), info

    @jax.jit
    def score_actions(self, observations, goals, actions):
        """Return conservative twin-Q scores for latent state-goal-action tuples."""
        q1, q2 = self.network.select('critic')(observations, goals, actions)
        return jnp.minimum(q1, q2)

    @classmethod
    def create(cls, seed, ex_latents, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
        )
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
        )
        network_info = {
            'value': (value_def, (ex_latents, ex_latents)),
            'critic': (critic_def, (ex_latents, ex_latents, ex_actions)),
            'target_critic': (
                copy.deepcopy(critic_def),
                (ex_latents, ex_latents, ex_actions),
            ),
        }
        network_def = ModuleDict({key: value[0] for key, value in network_info.items()})
        network_params = network_def.init(
            init_rng, **{key: value[1] for key, value in network_info.items()}
        )['params']
        network_params['modules_target_critic'] = network_params['modules_critic']
        network = TrainState.create(
            network_def,
            network_params,
            tx=optax.adam(config['lr']),
        )
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='gciql_chunk_lewm_shared',
            lr=3e-4,
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            chunk_size=5,
            latent_dim=192,
        )
    )
