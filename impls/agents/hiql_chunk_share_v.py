"""HIQL with a shared value representation and a chunk-conditioned low-level critic."""

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from agents.hiql import HIQLAgent
from agents.hiql import get_config as get_hiql_config
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCActor, Identity, LengthNormalize, MLP, ensemblize


class SharedGoalValue(nn.Module):
    """Twin goal-conditioned value with the single actor-visible goal representation."""

    hidden_dims: tuple
    layer_norm: bool
    state_encoder: nn.Module
    goal_encoder: nn.Module

    def setup(self):
        value_net = ensemblize(MLP, 2)(
            (*self.hidden_dims, 1),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        self.value_net = value_net

    def __call__(self, observations, goals, return_goal_rep=False):
        state_reps = self.state_encoder(observations)
        goal_reps = self.goal_encoder(jnp.concatenate([observations, goals], axis=-1))
        values = self.value_net(jnp.concatenate([state_reps, goal_reps], axis=-1)).squeeze(-1)
        if return_goal_rep:
            return values, goal_reps
        return values


class EncodedGoalCritic(nn.Module):
    """Twin chunk critic consuming the shared value representation of a goal."""

    hidden_dims: tuple
    layer_norm: bool
    state_encoder: nn.Module

    def setup(self):
        critic_net = ensemblize(MLP, 2)(
            (*self.hidden_dims, 1),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        self.critic_net = critic_net

    def __call__(self, observations, goal_reps, actions):
        state_reps = self.state_encoder(observations)
        inputs = jnp.concatenate([state_reps, goal_reps, actions], axis=-1)
        return self.critic_net(inputs).squeeze(-1)


class HIQLChunkShareVAgent(HIQLAgent):
    """Original HIQL shared value plus a GCIQL-style low-level chunk critic.

    Unlike :mod:`agents.hiql_chunk`, this agent has no independent low-level
    value.  One action-free HIQL value ``V(s, g)`` learns from the standard
    relabeled atomic transitions and supplies both the high-level actor and the
    low-level chunk critic.  The critic evaluates a demonstrated action chunk
    with a k-step target, and the low actor uses ``Q_chunk - V`` for AWR.
    """

    @property
    def action_horizon(self):
        return int(self.config['chunk_size'])

    def value_loss(self, batch, grad_params):
        goals = batch['value_goals']
        next_v1_t, next_v2_t = self.network.select('target_value')(batch['next_observations'], goals)
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        v1_t, v2_t = self.network.select('target_value')(batch['observations'], goals)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        values, goal_reps = self.network.select('value')(
            batch['observations'], goals, return_goal_rep=True, params=grad_params
        )
        v1, v2 = values
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2
        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'adv': adv.mean(),
            'goal_rep_batch_std': jnp.std(goal_reps, axis=0).mean(),
        }

    def low_critic_loss(self, batch, grad_params):
        next_v1, next_v2 = self.network.select('target_value')(
            batch['low_actor_next_observations'], batch['low_actor_goals']
        )
        next_v = jnp.minimum(next_v1, next_v2)
        chunk_discount = self.config['discount'] ** self.config['chunk_size']
        target_q = (
            batch['low_value_rewards']
            + chunk_discount * batch['low_value_masks'] * next_v
        )

        _, goal_reps = self.network.select('value')(
            batch['observations'], batch['low_actor_goals'], return_goal_rep=True
        )
        goal_reps = jax.lax.stop_gradient(goal_reps)
        q1, q2 = self.network.select('low_critic')(
            batch['observations'], goal_reps, batch['actions'], params=grad_params
        )
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': target_q.mean(),
            'q_max': target_q.max(),
            'q_min': target_q.min(),
            'target_std': target_q.std(),
            'chunk_reward_mean': batch['low_value_rewards'].mean(),
        }

    def low_actor_loss(self, batch, grad_params):
        values, frozen_goal_reps = self.network.select('value')(
            batch['observations'], batch['low_actor_goals'], return_goal_rep=True
        )
        v1, v2 = values
        v = (v1 + v2) / 2
        q1, q2 = self.network.select('low_critic')(
            batch['observations'], frozen_goal_reps, batch['actions']
        )
        q = jnp.minimum(q1, q2)
        adv = q - v
        weights = jnp.minimum(jnp.exp(adv * self.config['low_alpha']), 100.0)

        if self.config['low_actor_rep_grad']:
            _, actor_goal_reps = self.network.select('value')(
                batch['observations'],
                batch['low_actor_goals'],
                return_goal_rep=True,
                params=grad_params,
            )
        else:
            actor_goal_reps = frozen_goal_reps
        dist = self.network.select('low_actor')(
            batch['observations'], actor_goal_reps, goal_encoded=True, params=grad_params
        )
        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -(weights * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
            'weight_mean': weights.mean(),
            'weight_max': weights.max(),
            'weight_clip_ratio': (weights >= 100.0).mean(),
            'mode_batch_std': jnp.std(dist.mode(), axis=0).mean(),
        }

    def high_actor_loss(self, batch, grad_params):
        v1, v2 = self.network.select('value')(batch['observations'], batch['high_actor_goals'])
        nv1, nv2 = self.network.select('value')(
            batch['high_actor_targets'], batch['high_actor_goals']
        )
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v
        weights = jnp.minimum(jnp.exp(adv * self.config['high_alpha']), 100.0)

        dist = self.network.select('high_actor')(
            batch['observations'], batch['high_actor_goals'], params=grad_params
        )
        _, target = self.network.select('value')(
            batch['observations'], batch['high_actor_targets'], return_goal_rep=True
        )
        log_prob = dist.log_prob(target)
        actor_loss = -(weights * log_prob).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - target) ** 2),
            'std': jnp.mean(dist.scale_diag),
            'weight_mean': weights.mean(),
            'weight_max': weights.max(),
            'weight_clip_ratio': (weights >= 100.0).mean(),
            'mode_batch_std': jnp.std(dist.mode(), axis=0).mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        del rng
        info = {}
        losses = []
        for name, loss_fn in (
            ('value', self.value_loss),
            ('low_critic', self.low_critic_loss),
            ('low_actor', self.low_actor_loss),
            ('high_actor', self.high_actor_loss),
        ):
            loss, loss_info = loss_fn(batch, grad_params)
            losses.append(loss)
            info.update({f'{name}/{key}': value for key, value in loss_info.items()})
        return sum(losses), info

    @staticmethod
    def _target_update(network, tau):
        online_params = network.params['modules_value']
        target_params = network.params['modules_target_value']
        network.params['modules_target_value'] = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau), online_params, target_params
        )

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self._target_update(new_network, self.config['tau'])
        return self.replace(network=new_network, rng=new_rng), info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        if config['discrete']:
            raise ValueError('HIQLChunkShareVAgent currently supports continuous action spaces only.')

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_goals = ex_observations
        ex_goal_reps = jnp.zeros((*ex_observations.shape[:-1], config['rep_dim']))
        if ex_observations.ndim > 2:
            ex_goal_reps = jnp.zeros((ex_observations.shape[0], config['rep_dim']))

        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
        else:
            encoder_module = Identity

        def make_goal_encoder():
            sequence = []
            if config['encoder'] is not None:
                sequence.append(encoder_module())
            sequence.extend(
                [
                    MLP(
                        hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                        activate_final=False,
                        layer_norm=config['layer_norm'],
                    ),
                    LengthNormalize(),
                ]
            )
            return nn.Sequential(sequence)

        def make_value():
            return SharedGoalValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                state_encoder=encoder_module(),
                goal_encoder=make_goal_encoder(),
            )

        low_critic_def = EncodedGoalCritic(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            state_encoder=encoder_module(),
        )
        low_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=ex_actions.shape[-1],
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=GCEncoder(state_encoder=encoder_module()),
        )
        if config['encoder'] is not None:
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            high_actor_encoder_def = None
        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['rep_dim'],
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=high_actor_encoder_def,
        )

        network_info = {
            'value': (make_value(), (ex_observations, ex_goals)),
            'target_value': (make_value(), (ex_observations, ex_goals)),
            'low_critic': (low_critic_def, (ex_observations, ex_goal_reps, ex_actions)),
            'low_actor': (
                low_actor_def,
                {
                    'observations': ex_observations,
                    'goals': ex_goal_reps,
                    'goal_encoded': True,
                },
            ),
            'high_actor': (high_actor_def, (ex_observations, ex_goals)),
        }
        network_def = ModuleDict({name: item[0] for name, item in network_info.items()})
        network_args = {name: item[1] for name, item in network_info.items()}
        params = network_def.init(init_rng, **network_args)['params']
        params['modules_target_value'] = params['modules_value']
        network = TrainState.create(network_def, params, tx=optax.adam(config['lr']))
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = get_hiql_config()
    config.agent_name = 'hiql_chunk_share_v'
    config.dataset_class = 'HIQLChunkDataset'
    config.chunk_size = 5
    config.discrete = False
    return config
