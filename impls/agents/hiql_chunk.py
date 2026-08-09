from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCValue, Identity, LengthNormalize


class HIQLChunkAgent(flax.struct.PyTreeNode):
    """HIQL with a chunk-conditioned low-level IQL critic and Gaussian chunk actor.

    This is the Gaussian-policy HIQL-Chunk variant: the high-level latent planner follows
    HIQL, while the low level treats a length-k action sequence as one action.  No
    flow-matching modules are used.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the asymmetric squared expectile loss."""
        weight = jnp.where(adv >= 0, expectile, 1 - expectile)
        return weight * diff**2

    def high_value_loss(self, batch, grad_params):
        """Compute the c-step high-level IVL loss."""
        next_v1_t, next_v2_t = self.network.select('target_value')(
            batch['high_value_next_observations'], batch['value_goals']
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        discount = self.config['discount'] ** self.config['subgoal_steps']
        q = batch['rewards'] + discount * batch['masks'] * next_v_t

        v1_t, v2_t = self.network.select('target_value')(batch['observations'], batch['value_goals'])
        adv = q - (v1_t + v2_t) / 2

        q1 = batch['rewards'] + discount * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + discount * batch['masks'] * next_v2_t
        v1, v2 = self.network.select('value')(
            batch['observations'], batch['value_goals'], params=grad_params
        )
        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        v = (v1 + v2) / 2
        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def low_value_loss(self, batch, grad_params):
        """Fit V_L to an expectile of the dataset chunk Q-values."""
        q1, q2 = self.network.select('target_low_critic')(
            batch['observations'], batch['value_goals'], batch['action_chunks']
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select('low_value')(
            batch['observations'], batch['value_goals'], params=grad_params
        )
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def low_critic_loss(self, batch, grad_params):
        """Regress Q_L on the dataset-supported k-step target."""
        next_v = self.network.select('low_value')(
            batch['chunk_next_observations'], batch['value_goals']
        )
        discount = self.config['discount'] ** self.config['chunk_size']
        target_q = batch['rewards'] + discount * batch['masks'] * next_v

        q1, q2 = self.network.select('low_critic')(
            batch['observations'],
            batch['value_goals'],
            batch['action_chunks'],
            params=grad_params,
        )
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

        q = jnp.minimum(q1, q2)
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'target_q_mean': target_q.mean(),
        }

    def low_actor_loss(self, batch, grad_params):
        """Advantage-weighted Gaussian regression on complete action chunks."""
        v = self.network.select('low_value')(batch['observations'], batch['low_actor_goals'])
        q1, q2 = self.network.select('low_critic')(
            batch['observations'], batch['low_actor_goals'], batch['action_chunks']
        )
        q = jnp.minimum(q1, q2)
        adv = q - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['low_actor_goals']], axis=-1),
            params=grad_params,
        )
        if not self.config['low_actor_rep_grad']:
            goal_reps = jax.lax.stop_gradient(goal_reps)
        dist = self.network.select('low_actor')(
            batch['observations'], goal_reps, goal_encoded=True, params=grad_params
        )
        log_prob = dist.log_prob(batch['action_chunks'])
        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['action_chunks']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    def high_actor_loss(self, batch, grad_params):
        """Compute the HIQL high-level latent-subgoal AWR loss."""
        v1, v2 = self.network.select('value')(batch['observations'], batch['high_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['high_actor_targets'], batch['high_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

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
        """Compute the joint high-level and low-level objective."""
        info = {}

        losses = []
        for prefix, loss_fn in (
            ('high_value', self.high_value_loss),
            ('low_value', self.low_value_loss),
            ('low_critic', self.low_critic_loss),
            ('low_actor', self.low_actor_loss),
            ('high_actor', self.high_actor_loss),
        ):
            loss, loss_info = loss_fn(batch, grad_params)
            losses.append(loss)
            info.update({f'{prefix}/{key}': value for key, value in loss_info.items()})

        return sum(losses), info

    def target_update(self, network, module_name):
        """Polyak-update one target module after the optimizer step."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update all HIQL-Chunk modules and their target networks."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        self.target_update(new_network, 'low_critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Plan a latent subgoal and sample one flattened Gaussian action chunk."""
        high_seed, low_seed = jax.random.split(seed)
        high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)
        goal_reps = high_dist.sample(seed=high_seed)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])

        low_dist = self.network.select('low_actor')(
            observations, goal_reps, goal_encoded=True, temperature=temperature
        )
        action_chunks = low_dist.sample(seed=low_seed)
        return jnp.clip(action_chunks, -1, 1)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a Gaussian-policy HIQL-Chunk agent."""
        if config['discrete']:
            raise ValueError('HIQLChunkAgent currently supports continuous action spaces only.')

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]
        chunk_action_dim = action_dim * config['chunk_size']
        ex_action_chunks = jnp.zeros((ex_actions.shape[0], chunk_action_dim), dtype=ex_actions.dtype)

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

        if config['encoder'] is not None:
            value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            low_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            low_critic_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_low_critic_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            low_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            low_critic_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_low_critic_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = None

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=value_encoder_def,
        )
        target_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_value_encoder_def,
        )
        low_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=low_value_encoder_def,
        )
        low_critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=low_critic_encoder_def,
        )
        target_low_critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_low_critic_encoder_def,
        )
        low_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=chunk_action_dim,
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

        network_info = dict(
            goal_rep=(goal_rep_def, jnp.concatenate([ex_observations, ex_goals], axis=-1)),
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
            low_value=(low_value_def, (ex_observations, ex_goals)),
            low_critic=(low_critic_def, (ex_observations, ex_goals, ex_action_chunks)),
            target_low_critic=(target_low_critic_def, (ex_observations, ex_goals, ex_action_chunks)),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
        )
        networks = {key: value[0] for key, value in network_info.items()}
        network_args = {key: value[1] for key, value in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        network.params['modules_target_value'] = network.params['modules_value']
        network.params['modules_target_low_critic'] = network.params['modules_low_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Return the default HIQL-Chunk configuration."""
    return ml_collections.ConfigDict(
        dict(
            agent_name='hiql_chunk',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.5,
            low_alpha=3.0,
            high_alpha=3.0,
            subgoal_steps=25,
            chunk_size=5,
            rep_dim=10,
            low_actor_rep_grad=False,
            const_std=True,
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),
            dataset_class='HIQLChunkDataset',
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
