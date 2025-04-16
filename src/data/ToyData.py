import jax
import jax.numpy as jnp
import jax.random as jrandom
from src.data.Data import DataGenerator
from src.utils.KeyMonitor import KeyMonitor
from functools import partial
import math

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
    def __init__(self, landmark_num: int, radius: float, center: jnp.ndarray, flatten: bool = False, seed: int = 0):
        super().__init__()
        self.landmark_num = landmark_num
        self.radius = radius
        self.center = center
        self.key_monitor = KeyMonitor(seed)
        self.flatten = flatten

    @partial(jax.jit, static_argnums=(0, 1, 2))  # Make landmark_num static
    def _generate_data_internal(self, landmark_num: int, flatten: bool):
        return fibonacci_sphere_points(landmark_num, self.radius, self.center, flatten)

    def generate_data(self, landmark_num: int, batch_size: int):
        # 生成一个完整的球面点集
        sphere_points = self._generate_data_internal(landmark_num, self.flatten)
        
        # 创建一个批次数组，每个批次都包含相同的完整球面
        if self.flatten:
            result = jnp.tile(sphere_points[None, :, :], (batch_size, 1, 1))
        else:
            result = jnp.tile(sphere_points[None, :, :, :], (batch_size, 1, 1, 1))
        
        return result

@partial(jax.jit, static_argnums=(0,1,3))
def fibonacci_sphere_points(n_points, radius=1.0, center=jnp.array([0.0, 0.0, 0.0]), flatten: bool = False):
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
    n_grid = int(math.sqrt(n_points))
    if not flatten:
        return points.reshape(n_grid, n_grid, 3)
    else:
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
        theta = jnp.linspace(0, 2 * jnp.pi, grid_size)
        phi = jnp.linspace(0, jnp.pi, grid_size)
        theta_grid, phi_grid = jnp.meshgrid(theta, phi)
        theta_flat = theta_grid.flatten()
        phi_flat = phi_grid.flatten()
        R = radius  # Radius
        x = R * jnp.cos(theta_flat) * jnp.sin(phi_flat)
        y = R * jnp.sin(theta_flat) * jnp.sin(phi_flat)
        z = R * jnp.cos(phi_flat)
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

# spherical coordinate sampled data generator 
# all the data with shape (L, 2L-1, N) is sampled on the unit sphere with the feature channels N
# in this case, the feature channels are decart coordinates
class S2ManifoldDataGenerator:
    """
    Generate data on various manifolds with s2fft-compatible sampling.
    
    This class can generate data on different manifolds (sphere, cylinder, torus)
    with sampling schemes compatible with s2fft for spherical harmonic transforms.
    """
    
    def __init__(self, L: int, sampling: str = "mw", manifold_type: str = "sphere", 
                 radius: float = 1.0, height: float = 2.0, minor_radius: float = 0.5, 
                 major_radius: float = 2.0, width: float = 0.5, center: jnp.ndarray = jnp.array([0.0, 0.0, 0.0]), seed=0):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.key = jax.random.PRNGKey(seed)
        self.L = L
        self.sampling = sampling
        self.manifold_type = manifold_type
        self.radius = radius
        self.height = height
        self.minor_radius = minor_radius
        self.major_radius = major_radius
        self.width = width
        self.center = center

    def generate_sampling_grid(self, L, sampling='mw'):
        """
        Generate angular sampling grid based on the requested scheme.
        
        Args:
            L: Bandwidth/resolution parameter
            sampling: Sampling scheme ('mw', 'mwss', or 'dh')
            
        Returns:
            theta_grid: 2D grid of theta values
            phi_grid: 2D grid of phi values
            ntheta: Number of theta samples
            nphi: Number of phi samples
        """
        if sampling == 'mw':
            # McEwen & Wiaux sampling
            ntheta = L
            nphi = 2*L-1
            
            theta = jnp.linspace(0, jnp.pi, ntheta, endpoint=True)
            phi = jnp.linspace(0, 2*jnp.pi, nphi, endpoint=False)
            
        elif sampling == 'mwss':
            # McEwen & Wiaux Symmetric Sampling
            ntheta = L + 1
            nphi = 2*L
            
            theta = jnp.linspace(0, jnp.pi, ntheta, endpoint=True)
            phi = jnp.linspace(0, 2*jnp.pi, nphi, endpoint=False)
            
        elif sampling == 'dh':
            # Driscoll & Healy sampling
            ntheta = 2*L
            nphi = 2*L
            
            theta = jnp.linspace(0, jnp.pi, ntheta, endpoint=False) + jnp.pi/(2*ntheta)
            phi = jnp.linspace(0, 2*jnp.pi, nphi, endpoint=False)
            
        else:
            raise ValueError(f"Unsupported sampling scheme: {sampling}. Use 'mw', 'mwss', or 'dh'")
        
        # Create meshgrid
        phi_grid, theta_grid = jnp.meshgrid(phi, theta)
        
        return theta_grid, phi_grid, ntheta, nphi
    
    def sphere(self, theta_grid, phi_grid, radius=1.0, center=None):
        """
        Generate points on a sphere.
        
        Args:
            theta_grid: Grid of theta values
            phi_grid: Grid of phi values
            radius: Sphere radius
            center: Sphere center coordinates, default is origin [0,0,0]
            
        Returns:
            points: 3D Cartesian coordinates on the sphere
        """
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])
        
        # Convert spherical to Cartesian coordinates
        x = radius * jnp.sin(theta_grid) * jnp.cos(phi_grid)
        y = radius * jnp.sin(theta_grid) * jnp.sin(phi_grid)
        z = radius * jnp.cos(theta_grid)
        
        # Apply center offset
        x = x + center[0]
        y = y + center[1]
        z = z + center[2]
        
        # Combine coordinates
        points = jnp.stack([x, y, z], axis=-1)
        
        return points
    
    def cylinder(self, theta_grid, phi_grid, radius=1.0, height=2.0, center=None):
        """
        Generate points on a cylinder.
        
        Args:
            theta_grid: Grid of theta values (used as height parameter)
            phi_grid: Grid of phi values (used as angular parameter)
            radius: Cylinder radius
            height: Cylinder height
            center: Cylinder center coordinates, default is origin [0,0,0]
            
        Returns:
            points: 3D Cartesian coordinates on the cylinder
        """
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])
        
        # Map theta from [0, pi] to [-height/2, height/2]
        h = height * (theta_grid / jnp.pi - 0.5)
        
        # Convert cylindrical to Cartesian coordinates
        x = radius * jnp.cos(phi_grid)
        y = radius * jnp.sin(phi_grid)
        z = h
        
        # Apply center offset
        x = x + center[0]
        y = y + center[1]
        z = z + center[2]
        
        # Combine coordinates
        points = jnp.stack([x, y, z], axis=-1)
        
        return points
    
    def torus(self, theta_grid, phi_grid, major_radius=2.0, minor_radius=0.5, center=None):
        """
        Generate points on a torus.
        
        Args:
            theta_grid: Grid of theta values (poloidal angle)
            phi_grid: Grid of phi values (toroidal angle)
            major_radius: Distance from center of tube to center of torus
            minor_radius: Radius of the tube
            center: Torus center coordinates, default is origin [0,0,0]
            
        Returns:
            points: 3D Cartesian coordinates on the torus
        """
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])
        
        # Convert torus parameters to Cartesian coordinates
        x = (major_radius + minor_radius * jnp.cos(theta_grid)) * jnp.cos(phi_grid)
        y = (major_radius + minor_radius * jnp.cos(theta_grid)) * jnp.sin(phi_grid)
        z = minor_radius * jnp.sin(theta_grid)
        
        # Apply center offset
        x = x + center[0]
        y = y + center[1]
        z = z + center[2]
        
        # Combine coordinates
        points = jnp.stack([x, y, z], axis=-1)
        
        return points
    
    def mobius_strip(self, theta_grid, phi_grid, radius=2.0, width=0.5, center=None):
        """
        Generate points on a Möbius strip.
        
        Args:
            theta_grid: Grid of theta values
            phi_grid: Grid of phi values
            radius: Main radius of the Möbius strip
            width: Width of the strip
            center: Center coordinates, default is origin [0,0,0]
            
        Returns:
            points: 3D Cartesian coordinates on the Möbius strip
        """
        if center is None:
            center = jnp.array([0.0, 0.0, 0.0])
        
        # Remap theta to [0, 2π]
        u = phi_grid
        # Remap phi to [-width/2, width/2]
        v = width * (theta_grid / jnp.pi - 0.5)
        
        # Parametric equations for Möbius strip
        x = (radius + v * jnp.cos(u/2)) * jnp.cos(u)
        y = (radius + v * jnp.cos(u/2)) * jnp.sin(u)
        z = v * jnp.sin(u/2)
        
        # Apply center offset
        x = x + center[0]
        y = y + center[1]
        z = z + center[2]
        
        # Combine coordinates
        points = jnp.stack([x, y, z], axis=-1)
        
        return points
    
    def generate_data(self, L, batch_size=1, **kwargs):
        """
        Generate data on the specified manifold with s2fft-compatible sampling.
        
        Args:
            L: Bandwidth/resolution parameter
            manifold_type: Type of manifold ('sphere', 'cylinder', 'torus', 'mobius')
            sampling: Sampling scheme ('MW', 'MWSS', or 'DH')
            batch_size: Batch size
            **kwargs: Additional parameters for the specific manifold
                - radius, center for sphere
                - radius, height, center for cylinder
                - major_radius, minor_radius, center for torus
                - radius, width, center for mobius_strip
            
        Returns:
            points: Tensor of shape (batch_size, ntheta, nphi, 3) with 3D coordinates
        """
        # Generate appropriate sampling grid
        theta_grid, phi_grid, ntheta, nphi = self.generate_sampling_grid(L, self.sampling)
        
        # Generate points on the requested manifold
        if self.manifold_type == 'sphere':
            points = self.sphere(theta_grid, phi_grid, self.radius, self.center)
        elif self.manifold_type == 'cylinder':
            points = self.cylinder(theta_grid, phi_grid, self.radius, self.height, self.center)
        elif self.manifold_type == 'torus':
            points = self.torus(theta_grid, phi_grid, self.major_radius, self.minor_radius, self.center)
        elif self.manifold_type == 'mobius':
            points = self.mobius_strip(theta_grid, phi_grid, self.radius, self.width, self.center)
        else:
            raise ValueError(f"Unsupported manifold type: {self.manifold_type}")
        
        # Add batch dimension
        if batch_size > 1:
            points = jnp.tile(points[None, ...], (batch_size, 1, 1, 1))
        else:
            points = points[None, ...]
            
        return points