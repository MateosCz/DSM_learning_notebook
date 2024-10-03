from abc import ABC, abstractmethod
from typing import Callable, Any

import jax.numpy as jnp
import jax.random as jrandom
from collections.abc import Callable
from jax.typing import ArrayLike, DTypeLike
import jax


class SDE(ABC):

    @abstractmethod
    def drift_fn(self):
        pass

    @abstractmethod
    def diffusion_fn(self):
        pass


class Kunita_Lagrange_SDE(SDE):
    def __init__(self, sigma: DTypeLike, kappa: DTypeLike, X0: ArrayLike):
        super.__init__()
        self.sigma = sigma
        self.kappa = kappa
        self.X0 = X0

    def drift_fn(self):
        drift= lambda x: 0
        return drift

    def diffusion_fn(self):
        kernel_fn = lambda x, y: self.sigma*jnp.exp(-(jnp.linalg.norm(x-y)**2)/(self.kappa**2))
        Q_half = lambda x: self.partial_integrate(kernel_fn, 30, x)
        return Q_half
    
    def partial_integrate(self, fn, sample_num, x):
        '''
        partial integration of the kernel function, the parameter x is the point at which the kernel function is evaluated
        y is the point at which the kernel function is integrated, the integration is done over the interval [0,1]
        '''

        ys = jnp.linspace(0, 1, sample_num)
        dy = 1.0/sample_num
        return jnp.sum(fn(x, ys))*dy
