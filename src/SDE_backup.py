from abc import ABC, abstractmethod
from typing import Tuple
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
        

class Brownian_Motion_SDE(SDE):
    def __init__(self, dim: int, sigma: DTypeLike, x0: jnp.ndarray):
        self.dim = dim
        self.sigma = sigma
        self.noise_size = x0.shape[0]
    def drift_fn(self, x, t):
        return jnp.zeros_like(x)
    
    def diffusion_fn(self, x, t):
        return jnp.eye(x.shape[0]) * self.sigma
    
    def Sigma(self, x, t):
        return jnp.matmul(self.diffusion_fn(x, t), self.diffusion_fn(x, t).T)
    


class Kunita_Eulerian_SDE(SDE):
    def __init__(self, sigma: DTypeLike, kappa: DTypeLike, grid_dim: int, grid_num: int, grid_range: Tuple[float, float], x0: jnp.ndarray):
        self.sigma = sigma
        self.kappa = kappa
        self.grid_dim = grid_dim
        self.grid_num = grid_num
        self.grid_range = grid_range
        self.noise_size = grid_num ** 2
    @property
    def grid(self):
        '''
        generate the grid points for the kernel function, depende on the dimension of the grid
        '''
        grid_x = jnp.linspace(*self.grid_range, self.grid_num)
        grid_y = jnp.linspace(*self.grid_range, self.grid_num)
        grid_x, grid_y = jnp.meshgrid(grid_x, grid_y, indexing='xy')
        grid = jnp.stack([grid_x, grid_y], axis=-1)
        grid = grid.reshape(-1, 2)
        return grid
    
    def drift_fn(self, x, t):
        drift= lambda x, t: 0
        return drift(x, t)


    def diffusion_fn(self, x, t):
        def Q_half(x, t):
            kernel_fn = lambda x, y: self.sigma * jnp.exp(-0.5 * jnp.linalg.norm(x - y, axis=-1) ** 2 / self.kappa ** 2)            
            Q_half = jax.vmap(jax.vmap(kernel_fn, in_axes=(0, None)), in_axes=(None, 0))(self.grid, x)
            # the integral(simulated) happens when we do the matrix multiplication in the sde solver, so here we just return the kernel matrix
            return Q_half 
        return Q_half(x, t)


    
    def Sigma(self, x, t):
        return jnp.matmul(self.diffusion_fn(x, t), self.diffusion_fn(x, t).T)
    

    
class Ornstein_Uhlenbeck_SDE(SDE):
    def __init__(self, theta: DTypeLike, mu: DTypeLike, sigma: DTypeLike):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dim = mu.shape[0]

    def drift_fn(self, x, t):
        drift = lambda x, t: self.theta * (self.mu - x)
        return drift(x, t)
    
    def diffusion_fn(self, x, t):
        return self.sigma * jnp.eye(x.shape[0])
    
    def Sigma(self, x, t):
        return jnp.matmul(self.diffusion_fn(x, t), self.diffusion_fn(x, t).T)
        


class Kunita_Lagrange_SDE(SDE):
    def __init__(self, sigma: float, kappa: float, x0: jnp.ndarray):
        super().__init__()
        self.sigma = sigma
        self.kappa = kappa
        self.noise_size = x0.shape[0]
    def drift_fn(self, x, t):
        return jnp.zeros_like(x)    

    def diffusion_fn(self, x, t):
        def Q_half(x, t):
            # Ensure x is at least 2D
            if x.ndim == 1:
                x = x.reshape(1, -1)  # Add batch dimension if missing
            elif x.ndim == 2:
                pass
            else:
                raise ValueError(f"Input x should be 1D or 2D, got shape {x.shape}")
            
            # Define kernel function
            kernel_fn = lambda x: self.sigma * jnp.exp(-jnp.sum(jnp.square(x), axis=-1) / self.kappa ** 2)
            
            # Compute pairwise distances
            x_expanded = x[:, None, :]  # Shape: (N, 1, D)
            x_transposed = x[None, :, :]  # Shape: (1, N, D)
            dist = x_expanded - x_transposed  # Shape: (N, N, D)
            
            # Compute kernel
            kernel = kernel_fn(dist)
            
            return kernel

        return Q_half(x, t)
    
    def Sigma(self, x, t):
        return jnp.matmul(self.diffusion_fn(x, t), self.diffusion_fn(x, t).T)
    
'''
Time reversed SDE, depend on the original SDE, induced by the doob's h transform and
Kolmogorov's backward equation
'''
class Time_Reversed_SDE(SDE):
    def __init__(self, original_sde: SDE, score_fn: Callable[[jnp.ndarray, float], jnp.ndarray], total_time: float, dt: float):
        super().__init__()
        self.original_sde = original_sde
        self.score_fn = score_fn
        self.total_time = total_time
        self.dt = dt
        self.epsilon = 1e-5
        self.noise_size = original_sde.noise_size
    def compute_div_sigma(self, x: jnp.ndarray, t: float) -> jnp.ndarray:
        def div_sigma_single(x_i):
            def sigma_comp(i):
                sigma_i = lambda x: self.original_sde.diffusion_fn(x, t)[i]
                return jnp.trace(jax.jacfwd(sigma_i)(x_i))
            return jax.vmap(sigma_comp)(jnp.arange(x_i.shape[0]))
        
        return jax.vmap(div_sigma_single)(x)
    



    def drift_fn(self, x, t, x0):
        def drift_fn_impl(x,t, x0):
            drift = -self.original_sde.drift_fn(x, self.total_time - t) +\
                    jnp.matmul(self.original_sde.Sigma(x, self.total_time - t), self.score_fn(x, self.total_time - t, x0))
            div_sigma = self.compute_div_sigma(x, self.total_time - t)
            drift -= div_sigma
            return drift
 
        return drift_fn_impl(x, t, x0)
    
    def diffusion_fn(self, x, t):
        return self.original_sde.diffusion_fn(x, self.total_time - t)
    def Sigma(self, x, t):
        return jnp.matmul(self.diffusion_fn(x, t), self.diffusion_fn(x, t).T)
