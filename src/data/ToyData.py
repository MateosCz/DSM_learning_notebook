import jax
import jax.numpy as jnp

def generate_one_circle_data(landmarks: int, radius: float):
    theta = jnp.linspace(0, 2 * jnp.pi, landmarks)
    x = radius * jnp.cos(theta)
    y = radius * jnp.sin(theta)
    return jnp.stack([x, y], axis=-1)

def generate_circle_datas(batch_size: int, landmarks: int, radius: float):
    return jax.vmap(generate_one_circle_data)(jnp.arange(batch_size), landmarks, radius)