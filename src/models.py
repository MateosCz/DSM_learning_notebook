import jax
import jax.numpy as jnp
import flax.linen as nn


def get_timestep_embedding(t, embedding_dim=128):
    """
    Build sinusoidal embeddings for a single time value (adapted from Fairseq to JAX).
    
    Parameters
    ----------
    t : float or jnp.ndarray 
        Single time value
    embedding_dim : int 
        Dimension of time embedding
                        
    Returns
    -------    
    emb : jnp.ndarray
        Time embedding of shape (embedding_dim,)
    """
    
    scaling_factor = 100.0
    half_dim = embedding_dim // 2
    
    # Create embedding
    emb = jnp.log(10000.0) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = scaling_factor * emb * t
    
    # Compute sin and cos embeddings
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)])
    
    # Zero pad if needed
    if embedding_dim % 2 == 1:
        emb = jnp.pad(emb, (0, 1), mode='constant')
        
    return emb


class DsmModel(nn.Module):
    score_hidden_dims: tuple = (128,256,128)
    x_hidden_dims: tuple = (128,256,128)
    t_hidden_dims: tuple = (128,256,128)
    with_x0: bool = False
    dim: int = 2
    t_embedding_dim: int = 32
    @nn.compact
    def __call__(self, x, t, x0=None):
        t = get_timestep_embedding(t, self.t_embedding_dim)
        for hidden_dim in self.t_hidden_dims:
            t = nn.Dense(features=hidden_dim)(t)
            t = nn.gelu(t)
        if self.with_x0:
            x = jnp.concatenate([x, x0], axis=-1)
        for hidden_dim in self.x_hidden_dims:
            x = nn.Dense(features=hidden_dim)(x)
            x = nn.gelu(x)
        t= jnp.broadcast_to(jnp.expand_dims(t, axis=0), (x.shape[0], x.shape[-1]))
        x = jnp.concatenate([x, t], axis=-1)
        for hidden_dim in self.score_hidden_dims:
            x = nn.Dense(features=hidden_dim)(x)
            x = nn.gelu(x)
        x = nn.Dense(features=self.dim)(x)
        return x
    
