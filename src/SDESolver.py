import jax
import jax.numpy as jnp
import jax.random as jrandom
from collections.abc import Callable
from jaxtyping import Array, PyTree
from abc import ABC, abstractmethod

class SDESolver(ABC):
    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def step(self, t, x):
        pass

    @abstractmethod
    def renew_key(self, batch_size):
        pass

    @abstractmethod
    def sample_dW(self, key):
        pass



class EulerMaruyama(SDESolver):
    def __init__(self, drift_term: Callable[..., float], diffusion_term: Callable[..., float], x0, time_step, rng_key):
        self.drift_term = drift_term
        self.diffusion_term = diffusion_term
        self.x0 = x0

        self.time_step = time_step
        self.dim = x0.shape[-1]
        self.dt = 1.0/time_step
        self.batch_size = x0.shape[0]
        self.rng_key = rng_key

    def __call__(self):
        return self.solve()

    def solve(self):
        t = 0.0
        x = self.x0
        xs = []
        while t < 1.0:
            x = self.step(t, x)
            # print(x.shape)
            t = t + self.dt
            xs.append(x)
        return jnp.stack(xs, axis=0)

    def step(self, t, x):
        dt = self.dt
        key = self.renew_key(self.batch_size)
        dW = jax.vmap(self.sample_dW)(key)
        # print(dW.shape)
        drift = self.drift_term(t, x)
        diffusion = self.diffusion_term(t, x)
        x = x + drift * dt + diffusion * dW
        return x

    def renew_key(self, batch_size):
        self.rng_key, subkey = jrandom.split(self.rng_key)
        return jrandom.split(subkey, batch_size)
    def sample_dW(self, key):
        return jrandom.normal(key, (self.dim,)) * jnp.sqrt(self.dt)


