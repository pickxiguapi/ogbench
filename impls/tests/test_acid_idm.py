import jax
import jax.numpy as jnp

from acid_idm import ACIDInverseDynamicsFlow, sample_inverse_actions


def test_acid_idm_shapes_and_sampling():
    model = ACIDInverseDynamicsFlow(
        embed_dim=12,
        action_dim=10,
        model_dim=24,
        num_layers=2,
        num_heads=3,
        mlp_dim=48,
    )
    key = jax.random.PRNGKey(0)
    noisy = jnp.zeros((4, 10), dtype=jnp.float32)
    current = jnp.zeros((4, 12), dtype=jnp.float32)
    next_z = jnp.ones((4, 12), dtype=jnp.float32)
    times = jnp.ones((4,), dtype=jnp.float32)
    variables = model.init(key, noisy, current, next_z, times)
    velocity = model.apply(variables, noisy, current, next_z, times)
    assert velocity.shape == (4, 10)
    actions = sample_inverse_actions(
        model, variables['params'], current, next_z, key, num_steps=1
    )
    assert actions.shape == (4, 10)
    assert jnp.isfinite(actions).all()
