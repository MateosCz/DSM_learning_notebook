import jax
import jax.numpy as jnp
import jax.random as jrandom
from src.data.Data import DataGenerator

class CircleDataGenerator(DataGenerator):
    def __init__(self, landmark_num: int, radius: float, center: jnp.ndarray):
        self.landmark_num = landmark_num
        self.radius = radius
        self.center = center
    def generate_data(self, key: jnp.ndarray, batch_size: int):
        keys = jrandom.split(key, batch_size)
        return generate_circle_datas(keys, batch_size, self.landmark_num, self.radius, self.center)


def generate_one_circle_data(key: jnp.ndarray, landmark_num: int, radius: float, center: jnp.ndarray):
    theta = jrandom.uniform(key, (landmark_num,), minval=0, maxval=2 * jnp.pi)
    x = radius * jnp.cos(theta) + center[0]
    y = radius * jnp.sin(theta) + center[1]
    return jnp.stack([x, y], axis=-1)

def generate_circle_datas(keys: jnp.ndarray, batch_size: int, landmark_num: int, radius: float, center: jnp.ndarray):
    return jax.vmap(generate_one_circle_data, in_axes=(0, None, None, None))(keys, landmark_num, radius, center)

