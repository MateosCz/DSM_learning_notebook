import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import abc
from jax.lax import scan
import src.Losses as Losses
from src.SDE import SDE
from src.SDESolver import SDESolver
from typing import Optional
import jax.random as jrandom
from src.data.Data import DataGenerator
from src.utils.KeyMonitor import KeyMonitor
from tqdm import tqdm
from functools import partial

class Trainer(abc.ABC):
    model: nn.Module
    @abc.abstractmethod
    def train_state_init(self, model: nn.Module, lr: float = 1e-3, model_kwargs: dict = {}):
        pass

    @abc.abstractmethod
    def train_epoch(self, train_state: train_state.TrainState, 
                   data_generator: DataGenerator, sde: SDE, solver: SDESolver, batch_size: int):
        pass

    @abc.abstractmethod
    def train(self, train_state: train_state.TrainState, sde: SDE, solver: SDESolver, 
             data_generator: DataGenerator, epochs: int, batch_size: int):
        pass

class SsmTrainer(Trainer):
    def __init__(self, seed: int = 0):
        self.key_monitor = KeyMonitor(seed)

    def train_state_init(self, model: nn.Module, lr: float = 1e-3, model_kwargs: dict = {}):
        init_key = self.key_monitor.next_key()
        params = model.init(init_key, model_kwargs['x'], model_kwargs['t'], model_kwargs['x0'])
        tx = optax.adam(lr)
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    def _generate_batch(self, data_generator: DataGenerator, batch_size: int):
        key = self.key_monitor.next_key()
        return data_generator.generate_data(key, batch_size)

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def _train_step(self, train_state: train_state.TrainState, x0: jnp.ndarray, sde: SDE, solver: SDESolver, solver_key: jnp.ndarray, solve_keys: jnp.ndarray):
        # Create new solver instance with provided key
        solver = solver.from_sde(
            sde=sde,
            dt=solver.dt,
            total_time=solver.total_time,
            noise_size=x0.shape[1],
            dim=x0.shape[2],
            rng_key=solver_key
        )
        
        # Solve SDE for each sample with provided keys
        training_data, diffusion_history = jax.vmap(solver.solve)(x0, solve_keys)
        
        # Process data - make sure xs and times have matching dimensions
        num_timesteps = training_data.shape[1]
        times = jnp.linspace(0, solver.total_time, num_timesteps)
        xs = training_data
        
        # Compute Sigmas and drifts for all timesteps
        Sigmas = jax.vmap(jax.vmap(sde.Sigma, in_axes=(0, 0)), in_axes=(0, None))(xs, times)
        drifts = jax.vmap(jax.vmap(sde.drift_fn, in_axes=(0, 0)), in_axes=(0, None))(xs, times)

        def loss_fn(params):
            loss = jax.vmap(Losses.ssm_dsm_loss, in_axes=(None, None, 0, None, 0, 0, 0))(
                params, train_state, xs, times, x0, Sigmas, drifts
            )
            return jnp.mean(loss, axis=0)

        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss

    def train_epoch(self, train_state: train_state.TrainState, 
                   data_generator: DataGenerator, sde: SDE, solver: SDESolver, batch_size: int):
        x0 = self._generate_batch(data_generator, batch_size)
        solver_key = self.key_monitor.next_key()
        solve_keys = self.key_monitor.split_keys(x0.shape[0])
        return self._train_step(train_state, x0, sde, solver, solver_key, solve_keys)

    def train(self, train_state: train_state.TrainState, sde: SDE, solver: SDESolver, 
              data_generator: DataGenerator, epochs: int, batch_size: int):
        losses = jnp.zeros(epochs)
        
        for i in tqdm(range(epochs)):
            train_state, loss = self.train_epoch(train_state, data_generator, sde, solver, batch_size)
            losses = losses.at[i].set(loss)
        return train_state, losses
