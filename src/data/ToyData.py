import jax
import jax.numpy as jnp
import jax.random as jrandom
from src.data.Data import DataGenerator
from src.utils.KeyMonitor import KeyMonitor
from functools import partial
import igl

class CircleDataGenerator(DataGenerator):
    def __init__(self, landmark_num: int, radius: float, center: jnp.ndarray, seed: int = 0):
        super().__init__()
        self.landmark_num = landmark_num
        self.radius = radius
        self.center = center
        self.key_monitor = KeyMonitor(seed)

    @partial(jax.jit, static_argnums=(0,))
    def _generate_data_internal(self, keys: jnp.ndarray, landmark_num: int):
        return generate_circle_datas(keys, landmark_num, self.radius, self.center)

    def generate_data(self, landmark_num: int, batch_size: int):
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

    @partial(jax.jit, static_argnums=(0,2))
    def _generate_data_internal(self, keys: jnp.ndarray, landmark_num: int):
        return generate_ellipse_datas(keys, landmark_num, self.a, self.b, self.rotation_matrix, self.center)

    def generate_data(self, landmark_num: int, batch_size: int):
        keys = self.key_monitor.split_keys(batch_size)
        return self._generate_data_internal(keys, landmark_num)

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


class SphereDataGenerator(DataGenerator):
    def __init__(self, landmark_num: int, radius: float, center: jnp.ndarray, seed: int = 0):
        super().__init__()
        self.landmark_num = landmark_num
        self.radius = radius
        self.center = center
        self.key_monitor = KeyMonitor(seed)

    @partial(jax.jit, static_argnums=(0, 1))  # Make landmark_num static
    def _generate_data_internal(self, landmark_num: int):
        return fibonacci_sphere_points(landmark_num, self.radius, self.center)

    def generate_data(self, landmark_num: int, batch_size: int):
        # 生成一个完整的球面点集
        sphere_points = self._generate_data_internal(landmark_num)
        
        # 创建一个批次数组，每个批次都包含相同的完整球面
        result = jnp.tile(sphere_points[None, :, :], (batch_size, 1, 1))
        
        return result

@partial(jax.jit, static_argnums=(0,1))
def fibonacci_sphere_points(n_points, radius=1.0, center=jnp.array([0.0, 0.0, 0.0])):
    """Generate more evenly distributed points using Fibonacci spiral method."""
    # Constants for golden ratio calculation
    phi = jnp.pi * (3.0 - jnp.sqrt(5.0))
    
    # Create evenly spaced points from 0 to n_points-1
    i = jnp.arange(n_points)
    
    # Calculate y coordinates (latitude)
    y = 1 - (i / (n_points - 1)) * 2
    
    # Calculate radius at y
    radius_at_y = jnp.sqrt(1 - y * y) 
    
    # Calculate theta (longitude) based on golden ratio
    theta = phi * i
    
    # Convert to Cartesian coordinates
    x = radius_at_y * jnp.cos(theta)
    z = radius_at_y * jnp.sin(theta)
    
    # Stack coordinates and scale by radius
    points = jnp.column_stack([x, y, z]) * radius + center
    return points
class ManifoldDataGenerator(DataGenerator):
    def __init__(self, grid_size: int, manifold_type: str = "torus",flatten: bool = False, seed: int = 0):
        super().__init__()
        self.grid_size = grid_size
        self.manifold_type = manifold_type
        self.flatten = flatten
        self.key_monitor = KeyMonitor(seed)

    @partial(jax.jit, static_argnums=(0, 1, 2))  # Make grid_size and manifold_type static
    def _generate_data_internal(self, grid_size: int, manifold_type: str):
        return parametric_surface(grid_size, manifold_type)

    def generate_data(self, grid_size: int, batch_size: int):
        # 生成一个完整的流形点集
        manifold_points = self._generate_data_internal(grid_size, self.manifold_type)
        
        # 创建一个批次数组，每个批次都包含相同的完整流形
        result = jnp.tile(manifold_points[None, :, :], (batch_size, 1, 1))
        
        # reshape the result to (batch_size, grid_size * grid_size, 3)
        if self.flatten:
            return result.reshape(batch_size, grid_size * grid_size, 3)
        else:
            return result

@partial(jax.jit, static_argnums=(0, 1))
def parametric_surface(grid_size, manifold_type="torus"):
    """
    Generate points on a 2D manifold embedded in 3D space.
    
    Args:
        grid_size: Number of points in each parametric direction (u,v)
        manifold_type: Type of manifold ("torus", "cylinder", "mobius", etc.)
    
    Returns:
        Array of 3D points representing the manifold
    """
    # Create parameter grid
    u = jnp.linspace(0, 2 * jnp.pi, grid_size)
    v = jnp.linspace(0, 2 * jnp.pi, grid_size)
    u_grid, v_grid = jnp.meshgrid(u, v)
    
    # Flatten for easier processing
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()
    
    # Initialize coordinates array
    n_points = grid_size * grid_size
    points = jnp.zeros((n_points, 3))
    
    # Apply the appropriate parametrization based on manifold type
    if manifold_type == "torus":
        # Torus parameters
        R = 2.0  # Major radius
        r = 0.5  # Minor radius
        
        # Parametric equations for torus
        x = (R + r * jnp.cos(v_flat)) * jnp.cos(u_flat)
        y = (R + r * jnp.cos(v_flat)) * jnp.sin(u_flat)
        z = r * jnp.sin(v_flat)
        
    elif manifold_type == "cylinder":
        # Cylinder parameters
        R = 1.0  # Radius
        height = 2.0
        
        # Parametric equations for cylinder
        x = R * jnp.cos(u_flat)
        y = R * jnp.sin(u_flat)
        z = height * (v_flat / (2 * jnp.pi) - 0.5)
        
    elif manifold_type == "mobius":
        # Möbius strip parameters
        R = 2.0  # Major radius
        width = 0.5  # Width of the strip
        
        # Parametric equations for Möbius strip
        # Remap v to [-width/2, width/2]
        v_mapped = width * (v_flat / (2 * jnp.pi) - 0.5)
        
        x = (R + v_mapped * jnp.cos(u_flat/2)) * jnp.cos(u_flat)
        y = (R + v_mapped * jnp.cos(u_flat/2)) * jnp.sin(u_flat)
        z = v_mapped * jnp.sin(u_flat/2)
    
    elif manifold_type == "klein_bottle":
        # Klein bottle parameters
        R = 2.0
        
        # Parametric equations for Klein bottle (one immersion in 3D)
        # Remap parameters for easier equations
        u_mapped = u_flat * 2  # [0, 4π]
        v_mapped = v_flat      # [0, 2π]
        
        x = jnp.where(
            u_mapped < 2 * jnp.pi,
            (R + jnp.cos(v_mapped)) * jnp.cos(u_mapped),
            (R + jnp.cos(v_mapped)) * jnp.cos(u_mapped)
        )
        
        y = jnp.where(
            u_mapped < 2 * jnp.pi,
            (R + jnp.cos(v_mapped)) * jnp.sin(u_mapped),
            (R + jnp.cos(v_mapped)) * jnp.sin(u_mapped)
        )
        
        z = jnp.where(
            u_mapped < 2 * jnp.pi,
            jnp.sin(v_mapped),
            -jnp.sin(v_mapped)
        )
        
    else:  # Default to a simple plane
        # Plane parameters
        size = 2.0
        
        # Remap parameters to [-size, size]
        u_mapped = size * (u_flat / (2 * jnp.pi) - 0.5) * 2
        v_mapped = size * (v_flat / (2 * jnp.pi) - 0.5) * 2
        
        # Parametric equations for plane
        x = u_mapped
        y = v_mapped
        z = jnp.zeros_like(u_mapped)
    
    # Combine coordinates
    points = jnp.column_stack([x, y, z])
    return points

class ManifoldDataGenerator2D(DataGenerator):
    def __init__(self, grid_size: int, manifold_type: str = "torus", radius: float = 1.0, flatten: bool = False, seed: int = 0):
        super().__init__()
        self.grid_size = grid_size
        self.manifold_type = manifold_type
        self.radius = radius
        self.flatten = flatten
        self.key_monitor = KeyMonitor(seed)

    @partial(jax.jit, static_argnums=(0, 1, 2))  # Make grid_size and manifold_type static
    def _generate_data_internal(self, grid_size: int, manifold_type: str):
        return parametric_surface_2Dmanifold(grid_size, manifold_type, self.radius)

    def generate_data(self, grid_size: int, batch_size: int):
        # 生成一个完整的流形点集
        manifold_points = self._generate_data_internal(grid_size, self.manifold_type)
        
        # 重塑为 (grid_size, grid_size, 3) 以保持网格结构
        manifold_points = manifold_points.reshape(grid_size, grid_size, 3)
        
        # 创建一个批次数组，每个批次都包含相同的完整流形
        result = jnp.tile(manifold_points[None, :, :, :], (batch_size, 1, 1, 1))
        
        # flatten the 1,2 dimensions
        if self.flatten:
            return result.reshape(batch_size, grid_size * grid_size, 3)
        else:
            return result

@partial(jax.jit, static_argnums=(0, 1))
def parametric_surface_2Dmanifold(grid_size, manifold_type="torus", radius=1.0):
    """
    Generate points on a 2D manifold embedded in 3D space.
    
    Args:
        grid_size: Number of points in each parametric direction (u,v)
        manifold_type: Type of manifold ("torus", "cylinder", "mobius", etc.)
    
    Returns:
        Array of 3D points representing the manifold, shape (grid_size*grid_size, 3)
    """
    # Create parameter grid
    u = jnp.linspace(0, 2 * jnp.pi, grid_size)
    v = jnp.linspace(0, 2 * jnp.pi, grid_size)
    u_grid, v_grid = jnp.meshgrid(u, v)
    
    # Flatten for easier processing
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()
    
    # Initialize coordinates array
    n_points = grid_size * grid_size
    
    # Apply the appropriate parametrization based on manifold type
    if manifold_type == "sphere":
        # Sphere parameters
        R = radius  # Radius
        x = R * jnp.cos(u_flat) * jnp.sin(v_flat)
        y = R * jnp.sin(u_flat) * jnp.sin(v_flat)
        z = R * jnp.cos(v_flat)
    elif manifold_type == "torus":
        # Torus parameters
        R = radius  # Major radius
        r = 0.5  # Minor radius
        
        # Parametric equations for torus
        x = (R + r * jnp.cos(v_flat)) * jnp.cos(u_flat)
        y = (R + r * jnp.cos(v_flat)) * jnp.sin(u_flat)
        z = r * jnp.sin(v_flat)
        
    elif manifold_type == "cylinder":
        # Cylinder parameters
        R = radius  # Radius
        height = 2.0
        
        # Parametric equations for cylinder
        x = R * jnp.cos(u_flat)
        y = R * jnp.sin(u_flat)
        z = height * (v_flat / (2 * jnp.pi) - 0.5)
        
    elif manifold_type == "mobius":
        # Möbius strip parameters
        R = radius  # Major radius
        width = 0.5  # Width of the strip
        
        # Parametric equations for Möbius strip
        # Remap v to [-width/2, width/2]
        v_mapped = width * (v_flat / (2 * jnp.pi) - 0.5)
        
        x = (R + v_mapped * jnp.cos(u_flat/2)) * jnp.cos(u_flat)
        y = (R + v_mapped * jnp.cos(u_flat/2)) * jnp.sin(u_flat)
        z = v_mapped * jnp.sin(u_flat/2)
    
    elif manifold_type == "klein_bottle":
        # Klein bottle parameters
        R = radius
        
        # Parametric equations for Klein bottle (one immersion in 3D)
        # Remap parameters for easier equations
        u_mapped = u_flat * 2  # [0, 4π]
        v_mapped = v_flat      # [0, 2π]
        
        # 修正Klein bottle的参数方程
        condition = u_mapped < 2 * jnp.pi
        
        x = jnp.where(
            condition,
            (R + jnp.cos(v_mapped)) * jnp.cos(u_mapped),
            (R - jnp.cos(v_mapped)) * jnp.cos(u_mapped - 2 * jnp.pi)
        )
        
        y = jnp.where(
            condition,
            (R + jnp.cos(v_mapped)) * jnp.sin(u_mapped),
            (R - jnp.cos(v_mapped)) * jnp.sin(u_mapped - 2 * jnp.pi)
        )
        
        z = jnp.where(
            condition,
            jnp.sin(v_mapped),
            -jnp.sin(v_mapped)
        )
        
    else:  # Default to a simple plane
        # Plane parameters
        size = 2.0
        
        # Remap parameters to [-size, size]
        u_mapped = size * (u_flat / (2 * jnp.pi) - 0.5) * 2
        v_mapped = size * (v_flat / (2 * jnp.pi) - 0.5) * 2
        
        # Parametric equations for plane
        x = u_mapped
        y = v_mapped
        z = jnp.zeros_like(u_mapped)
    
    # Combine coordinates
    points = jnp.column_stack([x, y, z])
    return points
# class SphereDataGenerator:
#     """使用libigl和JAX生成球面数据的类"""
#     def __init__(self, landmark_num=32, radius=1.0, center=jnp.array([0.0, 0.0, 0.0]), seed=0):
#         self.landmark_num = landmark_num
#         self.radius = radius
#         self.center = center
#         self.key_monitor = KeyMonitor(seed)
        
#     def generate_regular_sphere(self):
#         """使用libigl生成规则球面网格"""
#         v, f = igl.sphere(self.landmark_num)
#         # 转换为JAX数组
#         vertices = jnp.array(v) * self.radius + self.center
#         faces = jnp.array(f)
#         return vertices, faces
    
#     @partial(jax.jit, static_argnums=(0, 2))
#     def _generate_data_internal(self, keys, landmark_num):
#         """内部方法，使用JAX生成球面数据"""
#         return generate_sphere_datas(keys, landmark_num, self.radius, self.center)
    
#     def generate_data(self, batch_size=1, use_igl=True):
#         """生成球面数据
        
#         Args:
#             batch_size: 要生成的球面数量
#             use_igl: 如果为True，使用libigl生成规则球面；否则使用JAX生成
            
#         Returns:
#             如果use_igl为True且batch_size为1，返回(vertices, faces)
#             否则返回一批球面顶点
#         """
#         if use_igl and batch_size == 1:
#             return self.generate_regular_sphere()
#         else:
#             keys = self.key_monitor.split_keys(batch_size)
#             return self._generate_data_internal(keys, self.landmark_num)
    
#     def generate_random_sphere(self, noise_level=0.1):
#         """生成带有随机噪声的球面
        
#         Args:
#             noise_level: 噪声的强度因子
            
#         Returns:
#             带有随机扰动的球面顶点和面
#         """
#         v, f = self.generate_regular_sphere()
#         key = self.key_monitor.split_keys(1)[0]
#         noise = jax.random.normal(key, shape=v.shape) * noise_level
#         noisy_v = v + noise
#         # 将顶点归一化回球面
#         norms = jnp.linalg.norm(noisy_v - self.center, axis=1, keepdims=True)
#         normalized_v = self.center + (noisy_v - self.center) / norms * self.radius
#         return normalized_v, f

# # 辅助函数
# @partial(jax.jit, static_argnums=(1))
# def generate_one_sphere_data(key, landmark_num, radius, center):
#     """生成单个球面的数据点"""
#     # 使用黄金螺旋算法生成均匀分布的点
#     indices = jnp.arange(0, landmark_num, dtype=float) + 0.5
#     phi = jnp.arccos(1 - 2 * indices / landmark_num)
#     theta = jnp.pi * (1 + 5**0.5) * indices
    
#     x = radius * jnp.cos(theta) * jnp.sin(phi) + center[0]
#     y = radius * jnp.sin(theta) * jnp.sin(phi) + center[1]
#     z = radius * jnp.cos(phi) + center[2]
    
#     # 添加一些随机扰动
#     noise = jax.random.normal(key, shape=(landmark_num, 3)) * 0.05 * radius
#     points = jnp.stack([x, y, z], axis=-1) + noise
    
#     # 将点归一化回球面
#     norms = jnp.linalg.norm(points - center, axis=1, keepdims=True)
#     normalized_points = center + (points - center) / norms * radius
    
#     return normalized_points

# @partial(jax.jit, static_argnums=(1))
# def generate_sphere_datas(keys, landmark_num, radius, center):
#     """批量生成多个球面数据"""
#     return jax.vmap(generate_one_sphere_data, in_axes=(0, None, None, None))(
#         keys, landmark_num, radius, center)