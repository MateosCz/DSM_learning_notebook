import jax.random as jrandom
import jax.numpy as jnp
from typing import Tuple

class KeyMonitor:
    def __init__(self, seed: int = 0):
        """Initialize with a seed for reproducibility"""
        self.main_key = jrandom.PRNGKey(seed)
    
    def next_key(self) -> jnp.ndarray:
        """Get next key and update main key"""
        self.main_key, subkey = jrandom.split(self.main_key)
        return subkey
    
    def split_keys(self, num: int) -> jnp.ndarray:
        """Split into multiple keys"""
        self.main_key, subkey = jrandom.split(self.main_key)
        return jrandom.split(subkey, num)

    def get_sde_solve_keys(self, batch_size: int, num_steps: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get keys for SDE solver
        Args:
            batch_size: number of samples in batch
            num_steps: number of time steps
        Returns:
            solver_key: key for solver initialization
            step_keys: keys for each step and sample, shape (batch_size, num_steps)
        """
        solver_key = self.next_key()
        # Get a key for each timestep of each sample
        step_keys = self.split_keys(batch_size * num_steps)
        # Reshape to (batch_size, num_steps, 2)
        step_keys = step_keys.reshape(batch_size, num_steps, 2)
        return solver_key, step_keys 