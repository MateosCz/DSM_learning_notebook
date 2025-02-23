import jax
import jax.numpy as jnp

@jax.jit
def get_rotation_matrix(theta):
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])

@jax.jit
def get_rotation_angle(rotation_matrix):
    return jnp.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
