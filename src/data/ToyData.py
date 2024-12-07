import jax
import jax.numpy as jnp
import jax.random as jrandom
from src.data.Data import DataGenerator
from src.utils.KeyMonitor import KeyMonitor
from functools import partial

class CircleDataGenerator(DataGenerator):
    def __init__(self, landmark_num: int, radius: float, center: jnp.ndarray, seed: int = 0):
        super().__init__()
        self.landmark_num = landmark_num
        self.radius = radius
        self.center = center
        self.key_monitor = KeyMonitor(seed)

    @partial(jax.jit, static_argnums=(0,))
    def _generate_data_internal(self, keys: jnp.ndarray):
        return generate_circle_datas(keys, self.landmark_num, self.radius, self.center)

    def generate_data(self, key: jnp.ndarray, batch_size: int):
        """Generate batch of data
        Args:
            key: PRNGKey for random number generation
            batch_size: number of samples to generate
        Returns:
            Array of shape (batch_size, landmark_num, 2)
        """
        # Ignore input key, use internal key monitor instead
        keys = self.key_monitor.split_keys(batch_size)
        return self._generate_data_internal(keys)

def generate_one_circle_data(key: jnp.ndarray, landmark_num: int, radius: float, center: jnp.ndarray):
    theta = jnp.linspace(0, 2 * jnp.pi, landmark_num)
    x = radius * jnp.cos(theta) + center[0]
    y = radius * jnp.sin(theta) + center[1]
    return jnp.stack([x, y], axis=-1)

def generate_circle_datas(keys: jnp.ndarray, landmark_num: int, radius: float, center: jnp.ndarray):
    return jax.vmap(generate_one_circle_data, in_axes=(0, None, None, None))(
        keys, landmark_num, radius, center)

