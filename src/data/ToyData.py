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

    def generate_data(self, batch_size: int):
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
    theta = jnp.linspace(0, 2 * jnp.pi, landmark_num+1)
    # theta_dist = theta[1] - theta[0]
    # # Generate random offsets for all points at once
    # random_offsets = jrandom.uniform(key, (landmark_num + 1,)) * theta_dist - theta_dist/2
    # # Add offsets to theta values
    # theta = theta + random_offsets

    x = radius * jnp.cos(theta) + center[0]
    y = radius * jnp.sin(theta) + center[1]
    circle_data = jnp.stack([x, y], axis=-1)
    circle_data = circle_data[:-1]
    return circle_data


def generate_circle_datas(keys: jnp.ndarray, landmark_num: int, radius: float, center: jnp.ndarray):
    return jax.vmap(generate_one_circle_data, in_axes=(0, None, None, None))(
        keys, landmark_num, radius, center)

class EllipseDataGenerator(DataGenerator):
    def __init__(self, landmark_num: int, a: float, b: float, rotation_matrix: jnp.ndarray, center: jnp.ndarray, seed: int = 0):
        super().__init__()
        self.landmark_num = landmark_num
        self.a = a
        self.b = b
        self.rotation_matrix = rotation_matrix
        self.center = center
        self.key_monitor = KeyMonitor(seed)

    @partial(jax.jit, static_argnums=(0,))
    def _generate_data_internal(self, keys: jnp.ndarray):
        return generate_ellipse_datas(keys, self.landmark_num, self.a, self.b, self.rotation_matrix, self.center)

    def generate_data(self, batch_size: int):
        keys = self.key_monitor.split_keys(batch_size)
        return self._generate_data_internal(keys)

def generate_one_ellipse_data(key: jnp.ndarray, landmark_num: int, a: float, b: float, rotation_matrix: jnp.ndarray, center: jnp.ndarray):
    theta = jnp.linspace(0, 2 * jnp.pi, landmark_num+1)
    x = a * jnp.cos(theta) + center[0]
    y = b * jnp.sin(theta) + center[1]
    ellipse_data = jnp.stack([x, y], axis=-1)
    ellipse_data = ellipse_data[:-1]
    ellipse_data = jax.vmap(lambda point: rotation_matrix @ point, in_axes=0)(ellipse_data)
    return ellipse_data

def generate_ellipse_datas(keys: jnp.ndarray, landmark_num: int, a: float, b: float, rotation_matrix: jnp.ndarray, center: jnp.ndarray):
    return jax.vmap(generate_one_ellipse_data, in_axes=(0, None, None, None, None, None))(
        keys, landmark_num, a, b, rotation_matrix,  center)