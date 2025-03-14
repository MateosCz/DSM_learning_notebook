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

    @partial(jax.jit, static_argnums=(0, 2))  # Make landmark_num static
    def _generate_data_internal(self, keys: jnp.ndarray, landmark_num: int):
        return jax.vmap(fibonacci_sphere_points, in_axes=(0, None, None, None))(keys, landmark_num, self.radius, self.center)

    def generate_data(self, landmark_num: int, batch_size: int):
        keys = self.key_monitor.split_keys(batch_size)
        return self._generate_data_internal(keys, landmark_num)

@partial(jax.jit, static_argnums=(1))  # Make landmark_num static
def generate_one_sphere_data(key: jnp.ndarray, landmark_num: int, radius: float, center: jnp.ndarray):
    theta = jnp.linspace(0, 2 * jnp.pi, landmark_num+1)
    phi = jnp.linspace(0, jnp.pi, landmark_num+1)
    x = radius * jnp.cos(theta) * jnp.sin(phi) + center[0]
    y = radius * jnp.sin(theta) * jnp.sin(phi) + center[1]
    z = radius * jnp.cos(phi) + center[2]
    sphere_v = jnp.stack([x, y, z], axis=-1)
    sphere_v = sphere_v[:-1]
    faces = []
    for i in range(landmark_num - 1):
        for j in range(landmark_num):
            j_next = (j + 1) % landmark_num
            v0 = i * landmark_num + j
            v1 = i * landmark_num + j_next
            v2 = (i + 1) * landmark_num + j
            v3 = (i + 1) * landmark_num + j_next
            faces.extend([[v0, v1, v2], [v2, v1, v3]])
    return sphere_v, jnp.array(faces)

@partial(jax.jit, static_argnums=(1))  # Make landmark_num static
def generate_sphere_datas(keys: jnp.ndarray, landmark_num: int, radius: float, center: jnp.ndarray):
    return jax.vmap(generate_one_sphere_data, in_axes=(0, None, None, None))(
        keys, landmark_num, radius, center)


@partial(jax.jit, static_argnums=(1,2))
def fibonacci_sphere_points(key: jnp.ndarray, n_points, radius=1.0, center=jnp.array([0.0, 0.0, 0.0])):
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