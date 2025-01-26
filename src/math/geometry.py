import jax
import jax.numpy as jnp

@jax.jit
def get_rotation_matrix(theta):
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])