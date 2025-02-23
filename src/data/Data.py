import abc
import jax.numpy as jnp
import jax.random as jrandom
import jax

class DataGenerator(abc.ABC):
    @abc.abstractmethod
    def generate_data(self, key: jnp.ndarray, batch_size: int):
        pass
