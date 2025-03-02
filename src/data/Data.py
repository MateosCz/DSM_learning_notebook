import abc
import jax.numpy as jnp
import jax.random as jrandom
import jax

class DataGenerator(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def generate_data(self, key: jnp.ndarray, batch_size: int):
        pass

    # @abc.abstractmethod
    # def visualize_samples(self, samples: jnp.ndarray):
    #     pass