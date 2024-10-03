from functools import partial
from jax import grad
import jax
import jax.numpy as jnp
import jax.random as jrandom

def OU_transition_density_log(x,t,x0,t0,theta,sigma):
    D = sigma**2/theta
    mean = x0 * jnp.exp(-theta*(t-t0))
    variance = D*(1-jnp.exp(-2*theta*(t-t0)))
    variance = jnp.eye(mean.shape[-1]) * variance
    pdf = jax.scipy.stats.multivariate_normal.pdf(x, mean, variance)
    return pdf