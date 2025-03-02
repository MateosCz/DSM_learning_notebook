import jax
import jax.numpy as jnp
import jax.random as jrandom
from src.data.Data import DataGenerator
from src.utils.KeyMonitor import KeyMonitor
from functools import partial

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

class ButterflyDataGenerator(DataGenerator):
    def __init__(self, original_landmarks, seed: int = 0):
        """
        Initialize the butterfly landmark generator using JAX
        Parameters:
        original_landmarks: array with shape (118, 2) containing the original landmark point coordinates
        """
        self.original_landmarks = jnp.array(original_landmarks)
        self.key_monitor = KeyMonitor(seed)
        # Check if the contour is closed; if not, add the first point to the end to close it
        is_closed = jnp.all(self.original_landmarks[0] == self.original_landmarks[-1])
        def true_branch(x):
            # Pad with a copy of the first point to match the shape of false_branch
            return jnp.vstack([x, x[0:1]])
        def false_branch(x):
            return jnp.vstack([x, x[0:1]])
        self.original_landmarks = jax.lax.cond(
            is_closed,
            true_branch,
            false_branch,
            self.original_landmarks
        )
        # Calculate cumulative distances (for contour parameterization)
        self.cumulative_distances = self._compute_cumulative_distances(self.original_landmarks)
        self.total_perimeter = self.cumulative_distances[-1]
        
        # Normalize parameters (within 0 to 1 range)
        self.normalized_params = self.cumulative_distances / self.total_perimeter
        
    def _compute_cumulative_distances(self, points):
        """Calculate cumulative distances between points using JAX"""
        # Calculate pairwise differences between consecutive points
        point_diffs = points[1:] - points[:-1]
        
        # Calculate Euclidean distances
        distances = jnp.sqrt(jnp.sum(point_diffs**2, axis=1))
        
        # Compute cumulative distances
        cumulative_dists = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(distances)])
        
        return cumulative_dists
    
    @partial(jax.jit, static_argnums=(0,))
    def _linear_interpolate(self, x, y, x_new):
        def interpolate_point(t):
            # Find indices where x[i] <= t < x[i+1]
            indices = jnp.sum(jnp.less_equal(x, t)) - 1
            indices = jnp.clip(indices, 0, x.shape[0] - 2)
            
            # Get surrounding points
            x1 = x[indices]
            x2 = x[indices + 1]
            y1 = y[indices]
            y2 = y[indices + 1]
            
            # Linear interpolation formula
            dx = x2 - x1
            # Handle the case where points might be identical (avoid division by zero)
            t_factor = jnp.where(dx > 1e-10, 
                                (t - x1) / dx,
                                0.0)
            
            # Compute interpolated value
            return y1 + t_factor * (y2 - y1)
        
        # Vectorize the interpolation function
        return jax.vmap(interpolate_point)(x_new)
    
    def _resample_landmarks(self, num_landmarks):
        """Resample points based on specified number of landmarks using JAX"""
        # Determine whether to include the end point
        is_closed = jnp.all(self.original_landmarks[0] == self.original_landmarks[-1])
        
        # Create a fixed number of points regardless of closed/open
        # Always use num_landmarks exactly
        new_params = jnp.linspace(0, 1, num_landmarks)
        
        # Perform cubic interpolation for x and y coordinates
        x_coords = self.original_landmarks[:, 0]
        y_coords = self.original_landmarks[:, 1]
        
        new_x = self._linear_interpolate(self.normalized_params, x_coords, new_params)
        new_y = self._linear_interpolate(self.normalized_params, y_coords, new_params)
        
        return jnp.column_stack((new_x, new_y))
    # def _resample_landmarks(self, num_landmarks):
    #     """Resample points based on specified number of landmarks using JAX"""
    #     # Determine whether to include the end point
    #     is_closed = jnp.all(self.original_landmarks[0] == self.original_landmarks[-1])
        
    #     # Create new parameterized values (evenly distributed)
    #     actual_num = jax.lax.cond(
    #         is_closed,
    #         lambda _: num_landmarks + 1,
    #         lambda _: num_landmarks,
    #         None
    #     )
        
    #     new_params = jnp.linspace(0, 1, actual_num)
        
    #     # For closed contour, remove the last point (duplicate of first point)
    #     new_params = jax.lax.cond(
    #         is_closed,
    #         lambda x: x[:-1],
    #         lambda x: x,
    #         new_params
    #     )
        
    #     # Perform cubic interpolation for x and y coordinates
    #     x_coords = self.original_landmarks[:, 0]
    #     y_coords = self.original_landmarks[:, 1]
        
    #     new_x = self._cubic_interpolate(self.normalized_params, x_coords, new_params)
    #     new_y = self._cubic_interpolate(self.normalized_params, y_coords, new_params)
        
    #     return jnp.column_stack((new_x, new_y))
    
    def _apply_transformation(self, landmarks, key, transform_params=None):
        """Apply transformations to landmarks (used for batch generation) using JAX"""
        if transform_params is None:
            # Split the key for different random operations
            key1, key2, key3, key4 = random.split(key, 4)
            
            # Randomly generate transformation parameters
            transform_params = {
                'rotation': random.uniform(key1, shape=(), minval=-0.1, maxval=0.1),  # Rotation angle (radians)
                'scale': random.uniform(key2, shape=(2,), minval=0.95, maxval=1.05),  # Scaling in x and y directions
                'translation': random.uniform(key3, shape=(2,), minval=-0.02, maxval=0.02),  # Translation in x and y
                'noise_level': 0.005,  # Noise level
                'noise_key': key4  # Key for noise generation
            }
            
        # Calculate center point
        center = jnp.mean(landmarks, axis=0)
        
        # Apply transformations
        # 1. Rotation
        angle = transform_params['rotation']
        rotation_matrix = jnp.array([
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)]
        ])
        
        # 2. Scaling
        scale = transform_params['scale']
        
        # 3. Translation
        ptp = jnp.max(jnp.ptp(landmarks, axis=0))
        translation = transform_params['translation'] * ptp
        
        # 4. Add noise
        if 'noise_key' in transform_params:
            noise = random.normal(transform_params['noise_key'], 
                                  shape=landmarks.shape) * transform_params['noise_level'] * ptp
        else:
            noise = jnp.zeros_like(landmarks)
        
        # Apply transformations
        transformed = landmarks - center  # Move to origin
        transformed = jnp.matmul(transformed, rotation_matrix.T)  # Rotate
        transformed = transformed * scale  # Scale
        transformed = transformed + center + translation + noise  # Move back and add translation and noise
        # don't apply transformation
        transformed = landmarks
        
        return transformed
    
    @partial(jax.jit, static_argnums=(0,3))
    def _generate_single(self, resampled, key, transform_params=None):
        """Generate a single sample (JIT-compiled for performance)"""
        return self._apply_transformation(resampled, key, transform_params)
    def generate_data(self, num_landmarks: int, batch_size: int, transform_params: dict = None):
        """
        Generate specified number of landmark points and batch size using JAX
        Parameters:
        num_landmarks: int, number of landmark points to generate
        batch_size: int, batch size to generate
        transform_params: dict or None, specifies transformation parameters. If None, random parameters are used
        
        Returns:
        JAX array with shape (batch_size, num_landmarks, 2)
        """
        # Create PRNG key
        keys = self.key_monitor.split_keys(batch_size)
        
        # Resample original landmarks
        resampled = self._resample_landmarks(num_landmarks)
        
        # Generate batch
        batch = jax.vmap(self._generate_single, in_axes=(None, 0, None))(resampled, keys, transform_params)
        return batch
    
    def visualize_samples(self, samples, figsize=(12, 8)):
        """Visualize generated samples"""
        samples_np = jnp.asarray(samples)
        
        batch_size = samples_np.shape[0]
        num_cols = min(4, batch_size)
        num_rows = (batch_size + num_cols - 1) // num_cols
        
        plt.figure(figsize=figsize)
        for i in range(batch_size):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.plot(samples_np[i, :, 0], samples_np[i, :, 1], 'b-o', markersize=2)
            plt.axis('equal')
            plt.title(f'Sample {i+1}')
            plt.grid(True)
            
        plt.tight_layout()
        plt.show()


