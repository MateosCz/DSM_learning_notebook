from meshplot import plot, subplot, interact
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import igl


def laplacian_beltrami(v, f):
    l = igl.cotmatrix(v, f)
    mass = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    m_inv = 1.0 / mass.diagonal()
    m_inv = jnp.diag(m_inv)
    laplacian = m_inv @ l
    return laplacian





