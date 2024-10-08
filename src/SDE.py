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


class Kunita_Eulerian_SDE(SDE):
    def __init__(self, sigma: DTypeLike, kappa: DTypeLike, grid_dim: int, grid_num: int, grid_range: Tuple[float, float] = (-0.5, 1.5)):
        self.sigma = sigma
        self.kappa = kappa
        self.grid_dim = grid_dim
        self.grid_num = grid_num
        self.grid_range = grid_range
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
    
    def drift_fn(self):
        drift= lambda x, t: 0
        return drift

    def diffusion_fn(self):
        def Q_half(x, t):
            kernel_fn = lambda x, y: self.sigma * jnp.exp(-0.5 * jnp.linalg.norm(x - y, axis=-1) ** 2 / self.kappa ** 2)            
            Q_half = jax.vmap(jax.vmap(kernel_fn, in_axes=(0, None)), in_axes=(None, 0))(self.grid, x)
            return Q_half
        return Q_half


    
class Ornstein_Uhlenbeck_SDE(SDE):
    def __init__(self, theta: DTypeLike, mu: DTypeLike, sigma: DTypeLike):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dim = mu.shape[0]

    def drift_fn(self):
        drift = lambda x, t: self.theta * (self.mu - x)
        return drift
    
    def diffusion_fn(self):
        return lambda x, t: self.sigma * jnp.eye(x.shape[0])
        


class Kunita_Lagrange_SDE(SDE):
    """ Eulerian SDE: dX(t) = Q^{1/2}(X(t)) dW(t),
        see ``Stochastic flows and shape bridges, S. Sommer et al.'' for details.
    """

    def __init__(self,
                 sigma: float = 1.0,
                 kappa: float = 0.1):
        super().__init__()
        self.sigma = sigma
        self.kappa = kappa
    

    def drift_fn(self):
        # no drift term for this SDE
        return lambda x, t: jnp.zeros_like(x)

    def diffusion_fn(self):
        """ Diffusion term of the Eulerian SDE defined by the Gaussian kernel k(x, y) = sigma * exp(-||x-y||^2 / kappa^2).
            The covariance is computed between the landmarks.
        """
        def Q_half(x, t):
            # make sure the input is 2D and has 2 columns
            assert x.ndim == 2 and x.shape[1] == 2, "Input x should have shape (n_pts, 2)"
            n_pts = x.shape[0]
            eps = 1e-4

            kernel_fn = lambda x: self.sigma * jnp.exp(-jnp.sum(jnp.square(x), axis=-1) / self.kappa ** 2)
            dist = x[:, None, :] - x[None, :, :]
            kernel = kernel_fn(dist) + eps * jnp.eye(n_pts)  # Regularization to avoid singularity
            # reshape the kernel to the desired output shape
            # Q_half = jnp.einsum("ij,kl->ijkl", kernel, jnp.eye(2))
            return kernel

        return Q_half