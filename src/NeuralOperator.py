import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from typing import Callable, List
import jax.scipy as jscipy
from tqdm.autonotebook import tqdm

class SpectralConv1d(eqx.Module):