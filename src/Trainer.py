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

class Trainer(abc.ABC):
    model: nn.Module
    @abc.abstractmethod
    def train_state_init(self, key: jnp.ndarray, lr: float = 1e-3, model_kwargs: dict = {}):
        pass

    @abc.abstractmethod
    def train_epoch(self, train_state: train_state.TrainState, data_generator: DataGenerator, sde: SDE, solver: SDESolver, batch_size: int):
        pass

    @abc.abstractmethod
    def train(self, train_state: train_state.TrainState, model: nn.Module, model_inputs: dict, epochs: int):
        pass

class SsmTrainer(Trainer):
    def train_state_init(self, key: jnp.ndarray, model: nn.Module, lr: float = 1e-3, model_kwargs: dict = {}):
        params =  model.init(key, model_kwargs['x'], model_kwargs['t'], model_kwargs['x0'])
        tx = optax.adam(lr)
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    def train_epoch(self, key: jnp.ndarray, train_state: train_state.TrainState, data_generator: DataGenerator, sde: SDE, solver: SDESolver, batch_size: int):
        # sample x0 each time
        key, subkey = jrandom.split(key)
        x0 = data_generator.generate_data(subkey, batch_size)
        key, subkey = jrandom.split(key)
        solver = solver.from_sde(sde, solver.dt, solver.total_time, data_generator.landmark_num, 2,subkey)
        key, subkey = jrandom.split(key)
        training_data, diffusion_history = jax.vmap(solver.solve, in_axes=(0, None))(x0, subkey)
        xs = training_data[:, 1:, :, :]
        times = jnp.linspace(0, solver.total_time, solver.num_steps)
        sigma_fn = sde.Sigma()
        drift_fn = sde.drift_fn()
        Sigmas = jax.vmap(sigma_fn, in_axes=(0, None))(xs, times)
        drifts = jax.vmap(drift_fn, in_axes=(0, None))(xs, times)
        def loss_fn(params):
            return jax.vmap(Losses.ssm_dsm_loss, in_axes=(None, None, 0, None, 0, 0, 0))(params, train_state, xs, times, x0, Sigmas, drifts)

        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss
    
    
    def train(self, train_state: train_state.TrainState, sde: SDE, solver: SDESolver, data_generator: DataGenerator, epochs: int, batch_size: int):
        key = jrandom.PRNGKey(0)
        losses = jnp.zeros(epochs)
        for i in range(epochs):
            key, subkey = jrandom.split(key)
            train_state, loss = self.train_epoch(subkey, train_state, data_generator, sde, solver, batch_size)
            losses = losses.at[i].set(loss)
        return train_state, losses
