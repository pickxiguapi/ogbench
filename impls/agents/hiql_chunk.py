"""Hierarchical IQL with a GCIQL-style fixed-length low-level action chunk."""

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from agents.hiql import HIQLAgent
from agents.hiql import get_config as get_hiql_config
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCActor, GCValue, Identity, LengthNormalize, MLP


class HIQLChunkAgent(HIQLAgent):
    """HIQL whose low level is GCIQL-Chunk AWR over continuous action chunks.

    The high-level planner keeps the original OGBench HIQL objective, but it
    has its own value ``V_H(s, g)``.  The low level mirrors GCIQL-Chunk: an
    action-conditioned twin critic ``Q_L(s, z, a[t:t + k])``, an expectile
    value ``V_L(s, z)``, and AWR weighted by ``min(Q_L) - V_L``.  Its critic
    uses the exact discounted k-step return and bootstraps from ``s[t + k]``
    with ``discount ** k``.
    """

    @property
    def action_horizon(self):
        """Number of atomic actions returned by one policy invocation."""
        return int(self.config['chunk_size'])

    def high_value_loss(self, batch, grad_params):
        """Compute the original one-step HIQL value objective for global goals."""
        goals = batch['value_goals']
        next_v1_t, next_v2_t = self.network.select('target_high_value')(batch['next_observations'], goals)
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        v1_t, v2_t = self.network.select('target_high_value')(batch['observations'], goals)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        v1, v2 = self.network.select('high_value')(batch['observations'], goals, params=grad_params)
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
        }

    def low_value_loss(self, batch, grad_params):
        """Fit V_L to an expectile of the target low-level chunk critic."""
        q1, q2 = self.network.select('target_low_critic')(
            batch['observations'], batch['low_actor_goals'], batch['actions']
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select('low_value')(
            batch['observations'], batch['low_actor_goals'], params=grad_params
        )
        adv = q - v
        value_loss = self.expectile_loss(adv, adv, self.config['low_expectile']).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'adv': adv.mean(),
        }

    def low_critic_loss(self, batch, grad_params):
        """Fit Q_L to the exact k-step return plus a V_L bootstrap."""
        next_v = self.network.select('low_value')(
            batch['low_actor_next_observations'], batch['low_actor_goals']
        )
        chunk_discount = self.config['discount'] ** self.config['chunk_size']
        target_q = batch['low_value_rewards'] + chunk_discount * batch['low_value_masks'] * next_v
        q1, q2 = self.network.select('low_critic')(
            batch['observations'], batch['low_actor_goals'], batch['actions'], params=grad_params
        )
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': target_q.mean(),
            'q_max': target_q.max(),
            'q_min': target_q.min(),
            'chunk_reward_mean': batch['low_value_rewards'].mean(),
        }

    def low_actor_loss(self, batch, grad_params):
        """Compute GCIQL-Chunk AWR on demonstrated low-level action chunks."""
        v = self.network.select('low_value')(batch['observations'], batch['low_actor_goals'])
        q1, q2 = self.network.select('low_critic')(
            batch['observations'], batch['low_actor_goals'], batch['actions']
        )
        q = jnp.minimum(q1, q2)
        adv = q - v

        exp_a = jnp.minimum(jnp.exp(adv * self.config['low_alpha']), 100.0)
        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['low_actor_goals']], axis=-1),
            params=grad_params,
        )
        if not self.config['low_actor_rep_grad']:
            goal_reps = jax.lax.stop_gradient(goal_reps)
        dist = self.network.select('low_actor')(
            batch['observations'], goal_reps, goal_encoded=True, params=grad_params
        )
        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    def high_actor_loss(self, batch, grad_params):
        """Compute the original HIQL high-level AWR objective using V_H."""
        v1, v2 = self.network.select('high_value')(batch['observations'], batch['high_actor_goals'])
        nv1, nv2 = self.network.select('high_value')(
            batch['high_actor_targets'], batch['high_actor_goals']
        )
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.minimum(jnp.exp(adv * self.config['high_alpha']), 100.0)
        dist = self.network.select('high_actor')(
            batch['observations'], batch['high_actor_goals'], params=grad_params
        )
        target = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['high_actor_targets']], axis=-1)
        )
        log_prob = dist.log_prob(target)
        actor_loss = -(exp_a * log_prob).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - target) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Update both value levels and both policy levels jointly."""
        del rng
        info = {}
        losses = []
        for name, loss_fn in (
            ('high_value', self.high_value_loss),
            ('low_value', self.low_value_loss),
            ('low_critic', self.low_critic_loss),
            ('low_actor', self.low_actor_loss),
            ('high_actor', self.high_actor_loss),
        ):
            loss, loss_info = loss_fn(batch, grad_params)
            losses.append(loss)
            info.update({f'{name}/{key}': value for key, value in loss_info.items()})
        return sum(losses), info

    @staticmethod
    def _target_update(network, module_name, tau):
        """EMA-update a target from post-optimizer online parameters."""
        target_name = f'target_{module_name}'
        online_params = network.params[f'modules_{module_name}']
        target_params = network.params[f'modules_{target_name}']
        network.params[f'modules_{target_name}'] = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau), online_params, target_params
        )

    @jax.jit
    def update(self, batch):
        """Apply Adam, then update the high value and low critic targets."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self._target_update(new_network, 'high_value', self.config['tau'])
        self._target_update(new_network, 'low_critic', self.config['tau'])
        return self.replace(network=new_network, rng=new_rng), info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create continuous hierarchical IQL with a GCIQL-Chunk low level."""
        if config['discrete']:
            raise ValueError('HIQLChunkAgent currently supports continuous action spaces only.')

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_goals = ex_observations

        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            goal_rep_seq = [encoder_module()]
        else:
            goal_rep_seq = []
        goal_rep_seq.extend(
            [
                MLP(
                    hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                    activate_final=False,
                    layer_norm=config['layer_norm'],
                ),
                LengthNormalize(),
            ]
        )
        goal_rep_def = nn.Sequential(goal_rep_seq)

        def make_value_encoder():
            if config['encoder'] is not None:
                return GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            return GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)

        def make_high_value():
            return GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=make_value_encoder(),
            )

        def make_low_value(*, ensemble):
            return GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=ensemble,
                gc_encoder=make_value_encoder(),
            )

        if config['encoder'] is not None:
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = None

        low_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=ex_actions.shape[-1],
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=low_actor_encoder_def,
        )
        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['rep_dim'],
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=high_actor_encoder_def,
        )

        network_info = {
            'goal_rep': (goal_rep_def, jnp.concatenate([ex_observations, ex_goals], axis=-1)),
            'high_value': (make_high_value(), (ex_observations, ex_goals)),
            'target_high_value': (make_high_value(), (ex_observations, ex_goals)),
            'low_value': (make_low_value(ensemble=False), (ex_observations, ex_goals)),
            'low_critic': (make_low_value(ensemble=True), (ex_observations, ex_goals, ex_actions)),
            'target_low_critic': (make_low_value(ensemble=True), (ex_observations, ex_goals, ex_actions)),
            'low_actor': (low_actor_def, (ex_observations, ex_goals)),
            'high_actor': (high_actor_def, (ex_observations, ex_goals)),
        }
        network_def = ModuleDict({name: item[0] for name, item in network_info.items()})
        network_args = {name: item[1] for name, item in network_info.items()}
        params = network_def.init(init_rng, **network_args)['params']
        params['modules_target_high_value'] = params['modules_high_value']
        params['modules_target_low_critic'] = params['modules_low_critic']
        network = TrainState.create(network_def, params, tx=optax.adam(config['lr']))
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Return standard HIQL defaults with a length-5 low-level chunk."""
    config = get_hiql_config()
    config.agent_name = 'hiql_chunk'
    config.dataset_class = 'HIQLChunkDataset'
    config.chunk_size = 5
    config.low_expectile = 0.9
    config.discrete = False
    return config
