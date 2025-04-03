from abc import ABC, abstractmethod
from typing import Tuple
from typing import Callable, Any

import jax.numpy as jnp
import jax.random as jrandom
from collections.abc import Callable
from jax.typing import ArrayLike, DTypeLike
import jax
from typing import Optional
from jax import vmap, jit, lax

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
        self.d_grid = ((grid_range[1]-grid_range[0]) / grid_num) ** 2 # small square grid size
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
            Q_half = jax.vmap(jax.vmap(kernel_fn, in_axes=(0, None)), in_axes=(None, 0))(self.grid, x) * self.d_grid
            # should we times a dy here?(or / grid_num)
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
            kernel = kernel_fn(dist)/self.noise_size
            
            return kernel

        return Q_half(x, t)
    
    def Sigma(self, x, t):
        return jnp.matmul(self.diffusion_fn(x, t), self.diffusion_fn(x, t).T)
    
'''
Time reversed SDE, depend on the original SDE, induced by the doob's h transform and
Kolmogorov's backward equation
'''
class Time_Reversed_SDE(SDE):
    def __init__(self, original_sde: SDE, score_fn: Callable[[jnp.ndarray, float], jnp.ndarray], total_time: float, dt: float, noise_size: Optional[int] = None):
        super().__init__()
        self.original_sde = original_sde
        self.score_fn = score_fn
        self.total_time = total_time
        self.dt = dt
        self.epsilon = 1e-5
        self.noise_size = noise_size if noise_size is not None else original_sde.noise_size
    def compute_div_sigma(self, x: jnp.ndarray, t: float) -> jnp.ndarray:
        def div_sigma_single(x_i):
            def sigma_comp(i):
                sigma_i = lambda x: self.original_sde.diffusion_fn(x, t)[i]
                return jnp.trace(jax.jacfwd(sigma_i)(x_i))
                # return jnp.trace(jax.jacrev(sigma_i)(x_i))
            return jax.vmap(sigma_comp)(jnp.arange(x_i.shape[0]))
        return jax.vmap(div_sigma_single)(x)



    def drift_fn(self, x, t, x0):
        jax.debug.print("score_fn: {0}", self.score_fn(x, self.total_time - t + self.dt, x0))
        def drift_fn_impl(x,t, x0):
            drift = -self.original_sde.drift_fn(x, self.total_time - t + self.dt) +\
                    jnp.matmul(self.original_sde.Sigma(x, self.total_time - t + self.dt), self.score_fn(x, self.total_time - t + self.dt, x0))
            div_sigma = self.compute_div_sigma(x, self.total_time - t + self.dt)
            drift -= div_sigma
            return drift
 
        return drift_fn_impl(x, t, x0)
    
    def diffusion_fn(self, x, t):
        return self.original_sde.diffusion_fn(x, self.total_time - t + self.dt)
    def Sigma(self, x, t):
        return jnp.matmul(self.diffusion_fn(x, t), self.diffusion_fn(x, t).T)


class Time_Reversed_SDE_2Dmanifold(SDE):
    def __init__(self, original_sde: SDE, score_fn: Callable[[jnp.ndarray, float], jnp.ndarray], total_time: float, dt: float, noise_size: Optional[int] = None):
        super().__init__()
        self.original_sde = original_sde
        self.score_fn = score_fn
        self.total_time = total_time
        self.dt = dt
        self.epsilon = 1e-5
        self.noise_size = noise_size if noise_size is not None else original_sde.noise_size
    def compute_div_sigma(self, x: jnp.ndarray, t: float) -> jnp.ndarray:
        def div_sigma_single(x_i):
            def sigma_comp(i):
                sigma_i = lambda x: self.original_sde.diffusion_fn(x, t)[i]
                return jnp.trace(jax.jacfwd(sigma_i)(x_i))
                # return jnp.trace(jax.jacrev(sigma_i)(x_i))
            return jax.vmap(sigma_comp)(jnp.arange(x_i.shape[0]))
        return jax.vmap(div_sigma_single)(x)



    def drift_fn(self, x, t, x0):
        jax.debug.print("score_fn: {0}", self.score_fn(x, self.total_time - t + self.dt, x0))
        def drift_fn_impl(x,t, x0):
            drift = -self.original_sde.drift_fn(x, self.total_time - t + self.dt) +\
                    jnp.einsum('ijkl,klm->ijm', self.original_sde.Sigma(x, self.total_time - t + self.dt), self.score_fn(x, self.total_time - t + self.dt, x0))
            # div_sigma = self.compute_div_sigma(x, self.total_time - t + self.dt)
            # drift -= div_sigma
            return drift
 
        return drift_fn_impl(x, t, x0)
    
    def diffusion_fn(self, x, t):
        return self.original_sde.diffusion_fn(x, self.total_time - t + self.dt)
    def Sigma(self, x, t):
        return jnp.einsum('ijk,klm->ijlm', self.diffusion_fn(x, t), self.diffusion_fn(x, t).T)

class Kunita_Flow_SDE_3D_Eulerian(SDE):
    '''
    Kunita flow sde in 3D, dx = sigma(x, t) * dW
    X: H: S -> R^3
    x dimension : (num_particles, 3) (R^d landmark position, d=3)
    t dimension : (num_particles, 1) (time)
    dw dimension : (num_particles, noize_size) (R^J wiener process, J = noize_size)
    sigma dimension : (num_particles, 3, noize_size) (R^d x R^J matrix)
    sigma(x, t) = kernel_fn(x, grid) * d_grid

    '''
    def __init__(self, k_alpha: DTypeLike, k_sigma: DTypeLike, grid_num: int, grid_range: Tuple[float, float], x0: jnp.ndarray):
        super().__init__()
        self.k_alpha = k_alpha
        self.k_sigma = k_sigma
        self.grid_dim = 3
        self.grid_num = grid_num
        self.grid_range = grid_range
        self.noise_size = grid_num ** 3
        self.d_grid = ((grid_range[1]-grid_range[0]) / grid_num) ** 3 # small square grid size

    @property
    def grid(self):
        grid_x = jnp.linspace(*self.grid_range, self.grid_num)
        grid_y = jnp.linspace(*self.grid_range, self.grid_num)
        grid_z = jnp.linspace(*self.grid_range, self.grid_num)
        grid_x, grid_y, grid_z = jnp.meshgrid(grid_x, grid_y, grid_z, indexing='xy')
        grid = jnp.stack([grid_x, grid_y, grid_z], axis=-1)
        grid = grid.reshape(-1, 3)
        return grid

    def drift_fn(self, x, t):
        return jnp.zeros_like(x)    

    def diffusion_fn(self, x, t):
        def Q_half(x, t):

            # define the kernel function
            kernel_fn = lambda x, y: self.k_alpha * jnp.exp(-0.5 * jnp.linalg.norm(x - y, axis=-1) ** 2 / self.k_sigma ** 2)
            # compute the kernel matrix
            print(self.grid.shape)
            print(x.shape)
            Q_half = jax.vmap(jax.vmap(kernel_fn, in_axes=(0, None)), in_axes=(None, 0))(self.grid, x) * self.d_grid
            # should we times a dy here?(or / grid_num)
            # the integral(simulated) happens when we do the matrix multiplication in the sde solver, so here we just return the kernel matrix
            return Q_half 
        return Q_half(x, t)
    
    def Sigma(self, x, t):
        return jnp.matmul(self.diffusion_fn(x, t), self.diffusion_fn(x, t).T)


class Kunita_Flow_SDE_3D_Eulerian_Optimized(SDE):
    '''
    优化版本的Kunita flow SDE in 3D, dx = sigma(x, t) * dW
    使用批处理和内存高效的算法来减少内存使用
    
    x dimension : (num_particles, 3) (R^d landmark position, d=3)
    t dimension : (num_particles, 1) (time)
    dw dimension : (num_particles, noize_size) (R^J wiener process, J = noize_size)
    sigma dimension : (num_particles, 3, noize_size) (R^d x R^J matrix)
    sigma(x, t) = kernel_fn(x, grid) * d_grid
    '''
    def __init__(self, k_alpha: DTypeLike, k_sigma: DTypeLike, grid_num: int, 
                 grid_range: Tuple[float, float], x0: jnp.ndarray, batch_size: int = 1024):
        super().__init__()
        self.k_alpha = k_alpha
        self.k_sigma = k_sigma
        self.grid_dim = 3
        self.grid_num = grid_num
        self.grid_range = grid_range
        self.noise_size = grid_num ** 3
        self.d_grid = ((grid_range[1]-grid_range[0]) / grid_num) ** 3  # 小立方体网格体积
        self.batch_size = batch_size  # 批处理大小
        
        # # 预编译一些函数以提高性能
        # self._compute_kernel = jit(self._kernel_function)
        # self._batched_kernel = jit(vmap(self._kernel_function, in_axes=(0, None)))

    @property
    def grid(self):

        grid_x = jnp.linspace(*self.grid_range, self.grid_num)
        grid_y = jnp.linspace(*self.grid_range, self.grid_num)
        grid_z = jnp.linspace(*self.grid_range, self.grid_num)
        grid_x, grid_y, grid_z = jnp.meshgrid(grid_x, grid_y, grid_z, indexing='xy')
        grid = jnp.stack([grid_x, grid_y, grid_z], axis=-1)
        grid = grid.reshape(-1, 3)
        return grid

    def drift_fn(self, x, t):
        return jnp.zeros_like(x)

    def _kernel_function(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """计算单对点之间的核函数值"""
        # x: (3,), y: (3,)
        return self.k_alpha * jnp.exp(-0.5 * jnp.sum((x - y) ** 2) / (self.k_sigma ** 2))

    def _diffusion_batch(self, x_batch: jnp.ndarray, grid_batch: jnp.ndarray) -> jnp.ndarray:
        """批量计算扩散函数的一部分"""
        # x_batch: (batch_size_x, 3), grid_batch: (batch_size_grid, 3)
        # 返回: (batch_size_x, batch_size_grid)
        return vmap(lambda x: vmap(lambda g: self._kernel_function(x, g))(grid_batch))(x_batch)

    def diffusion_fn(self, x, t):
        """
        优化的扩散函数，使用批处理来减少内存使用
        
        参数:
            x: 位置，形状为 (num_particles, 3)
            t: 时间，形状为 (num_particles, 1)
            
        返回:
            Q_half: 形状为 (num_particles, noise_size)
        """
        num_particles = x.shape[0]
        grid = self.grid  # (grid_num^3, 3)
        
        # 初始化结果矩阵
        Q_half = jnp.zeros((num_particles, self.noise_size))
        
        # 批处理粒子
        for i in range(0, num_particles, self.batch_size):
            end_i = min(i + self.batch_size, num_particles)
            x_batch = x[i:end_i]  # (batch_size_x, 3)
            
            # 为当前粒子批次构建Q_half的部分
            batch_result = jnp.zeros((end_i - i, self.noise_size))
            
            # 批处理网格点
            for j in range(0, self.noise_size, self.batch_size):
                end_j = min(j + self.batch_size, self.noise_size)
                grid_batch = grid[j:end_j]  # (batch_size_grid, 3)
                
                # 计算当前批次的核函数值
                kernel_values = self._diffusion_batch(x_batch, grid_batch)  # (batch_size_x, batch_size_grid)
                
                # 更新结果 - 确保形状匹配
                batch_result = batch_result.at[:, j:end_j].set(kernel_values)
            
            # 将当前批次的结果放入完整结果中
            Q_half = Q_half.at[i:end_i].set(batch_result * self.d_grid)
        
        return Q_half

    def Sigma(self, x, t):
        return jnp.matmul(self.diffusion_fn(x, t), self.diffusion_fn(x, t).T)


class Kunita_Flow_SDE_3D_Eulerian_2Dmanifold(SDE):
    '''
    2D manifold上的Kunita flow SDE, dx = sigma(x, t) * dW
    X: S1 x S1 -> R^3
    x dimension : (S1, S1, 3) (R^d landmark position, d=3)
    t dimension : (num_particles, 1) (time)
    '''
    def __init__(self, k_alpha: DTypeLike, k_sigma: DTypeLike, grid_num: int, grid_range: Tuple[float, float], x0: jnp.ndarray):
        super().__init__()
        self.k_alpha = k_alpha
        self.k_sigma = k_sigma
        self.grid_dim = 3
        self.grid_num = grid_num
        self.grid_range = grid_range
        self.noise_size = grid_num ** 3
        self.d_grid = ((grid_range[1]-grid_range[0]) / grid_num) ** 3 # small square grid size

    @property
    def grid(self):
        grid_x = jnp.linspace(*self.grid_range, self.grid_num)
        grid_y = jnp.linspace(*self.grid_range, self.grid_num)
        grid_z = jnp.linspace(*self.grid_range, self.grid_num)
        grid_x, grid_y, grid_z = jnp.meshgrid(grid_x, grid_y, grid_z, indexing='xy')
        grid = jnp.stack([grid_x, grid_y, grid_z], axis=-1)
        grid = grid.reshape(-1, 3)
        return grid

    def drift_fn(self, x, t):
        return jnp.zeros_like(x)    

    def diffusion_fn(self, x, t):
        def Q_half(x, t):

            # define the kernel function
            kernel_fn = lambda x, y: self.k_alpha * jnp.exp(-0.5 * jnp.linalg.norm(x - y, axis=-1) ** 2 / self.k_sigma ** 2)
            # compute the kernel matrix
            print(self.grid.shape)
            print(x.shape)
            Q_half = jax.vmap(jax.vmap(jax.vmap(kernel_fn, in_axes=(0, None)), in_axes=(None, 0)), in_axes=(None, 0))(self.grid, x) * self.d_grid

            # the integral(simulated) happens when we do the matrix multiplication in the sde solver, so here we just return the kernel matrix
            return Q_half 
        return Q_half(x, t)
    
    def Sigma(self, x, t):
        sigma = self.diffusion_fn(x, t)
        return jnp.einsum('ijk,klm->ijlm', sigma, sigma.T)


# Function to extract leaf butterfly landmarks
def extract_leaf_butterfly_landmarks(landmarks):
    """
    Extract butterfly landmarks without nodes that start with 'xx'
    
    Args:
        landmarks: DataFrame containing butterfly landmarks with 'node_names' column
        
    Returns:
        DataFrame with only leaf butterfly landmarks (no 'xx' prefixed nodes)
    """
    # Create a mask for rows where node_names don't start with 'xx'
    mask = ~landmarks['node_names'].str.startswith('xx')
    
    # Return filtered DataFrame
    return landmarks[mask].reset_index(drop=True)

