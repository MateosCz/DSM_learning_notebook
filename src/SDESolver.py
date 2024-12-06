import jax
import jax.numpy as jnp
import jax.random as jrandom
from collections.abc import Callable
from jaxtyping import Array, PyTree
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from src.SDE import SDE
# class SDESolver(ABC):
#     @abstractmethod
#     def __call__(self):
#         pass

#     @abstractmethod
#     def solve(self):
#         pass

#     @abstractmethod
#     def step(self, t, x):
#         pass

#     @abstractmethod
#     def renew_key(self, batch_size):
#         pass

#     @abstractmethod
#     def sample_dW(self, key):
#         pass
class SDESolver(ABC):
    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def from_sde(self, sde: SDE, x0: jnp.ndarray, dt: float, total_time: float, batch_size: int, rng_key: jnp.ndarray, x0_list: Optional[jnp.ndarray] = None, debug_mode: bool = False) -> 'SDESolver':
        pass



# class EulerMaruyama(SDESolver):
#     def __init__(self, drift_term: Callable[..., float], diffusion_term: Callable[..., float], x0, time_step, rng_key):
#         self.drift_term = drift_term # the drift term should be applied on one sample at a time, the batched function is not allowed
#         self.diffusion_term = diffusion_term # the diffusion term should be applied on one sample at a time, the batched function is not allowed
#         self.x0 = x0

#         self.time_step = time_step
#         self.dim = x0.shape[-1]
#         self.dt = 1.0/time_step
#         self.batch_size = x0.shape[0]
#         self.rng_key = rng_key

#     def __call__(self):
#         return self.solve()

#     def solve(self):
#         t = 0.0
#         x = self.x0
#         xs = []
#         xs.append(x)
#         t = t + self.dt
#         while t < 1.0:
#             key = self.renew_key(self.batch_size) 
#             x = jax.vmap(self.step, in_axes=(0,None,0))(x, t, key) # do step for each sample in the batch
#             t = t + self.dt
#             xs.append(x)
#         return jnp.stack(xs, axis=0)

#     def step(self, x, t, key):
#         dt = self.dt
#         dW = self.sample_dW(key)
#         drift = self.drift_term(x, t)
#         diffusion = self.diffusion_term(x, t)
#         x = x + drift * dt + jnp.dot(diffusion, dW)
#         return x

#     def renew_key(self, batch_size):
#         self.rng_key, subkey = jrandom.split(self.rng_key)
#         return jrandom.split(subkey, batch_size)

#     # def renew_key(self, dim):
#     #     self.rng_key, subkey = jrandom.split(self.rng_key, dim)
#     #     return subkey  
    
#     def sample_dW(self, key):
#         return jrandom.multivariate_normal(key, jnp.zeros(self.dim), jnp.eye(self.dim) * jnp.sqrt(self.dt))

class EulerMaruyama:
    def __init__(self, 
                 drift_fn: Callable[..., jnp.ndarray],
                 diffusion_fn: Callable[..., jnp.ndarray],
                 dt: float,
                 total_time: float,
                 noise_size: int,
                 dim: int,
                 rng_key: jnp.ndarray,
                 condition_x: Optional[jnp.ndarray] = None,
                 debug_mode: bool = False):
        self.drift_fn = drift_fn
        self.diffusion_fn = diffusion_fn
        self.dt = dt
        self.total_time = total_time
        self.num_steps = int(total_time / dt)
        self.rng_key = rng_key
        self.noise_size = noise_size
        self.condition_x = condition_x
        self.debug_mode = debug_mode
        self.dim = dim
    def solve(self, x0: jnp.ndarray, rng_key: jnp.ndarray) -> jnp.ndarray:
        def step(carry: Tuple[jnp.ndarray, jnp.ndarray], t: float):
            x, key = carry
            key, subkey = jrandom.split(key)
            subkey = jrandom.split(subkey, self.noise_size) # noise size normally is same as the x0 resolution, but not necessarily
            dW = jax.vmap(lambda key: jrandom.normal(key, (self.dim,)) * jnp.sqrt(self.dt), in_axes=(0))(subkey)
            # dW = jax.vmap(lambda key: jrandom.multivariate_normal(key, jnp.zeros(self.dim), jnp.eye(self.dim) * self.dt), in_axes=(0))(subkey)
            if self.condition_x is not None:
                drift = self.drift_fn(x,t, self.condition_x)    
            else:
                drift = self.drift_fn(x, t)
            diffusion = self.diffusion_fn(x, t)
            x_next = x + drift * self.dt + jnp.einsum('ij,jk->ik', diffusion, dW)
            if self.debug_mode:
                jax.debug.print("t: {t}", t=t)
                jax.debug.print("dt: {dt}", dt=self.dt)
                jax.debug.print("num_steps: {num_steps}", num_steps=self.num_steps)
                jax.debug.print("drift: {drift}", drift=drift)
                jax.debug.print("diffusion: {diffusion}", diffusion=diffusion)
                jax.debug.print("dW: {dW}", dW=dW)
                jax.debug.print("x_next: {x_next}", x_next=x_next)
            return (x_next, key), (x_next, diffusion)
        times = jnp.linspace(0, self.total_time, self.num_steps + 1)
        _, (trajectory, diffusion_history) = jax.lax.scan(step, (x0, rng_key), times[:-1])
        return jnp.concatenate([x0[None, ...], trajectory], axis=0), diffusion_history

    @staticmethod
    def from_sde(sde, dt: float, total_time: float, noise_size: int, dim: int, rng_key: jnp.ndarray, condition_x: Optional[jnp.ndarray] = None, debug_mode: bool = False) -> 'EulerMaruyama':
        return EulerMaruyama(sde.drift_fn(), sde.diffusion_fn(), dt, total_time, noise_size, dim, rng_key, condition_x, debug_mode)
