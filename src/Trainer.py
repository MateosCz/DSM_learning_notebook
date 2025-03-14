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
from tqdm import trange
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
        self.object_fn = "Heng"
    def train_state_init(self, model: nn.Module, lr: float = 1e-3, model_kwargs: dict = {}):
        init_key = self.key_monitor.next_key()
        params = model.init(init_key, model_kwargs['x'], model_kwargs['t'], model_kwargs['x0'])
        if 'object_fn' in model_kwargs:
            self.object_fn = model_kwargs['object_fn']
        tx = optax.adam(lr)
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    def _generate_batch(self, data_generator: DataGenerator, batch_size: int):
        return data_generator.generate_data(batch_size)

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def _train_step(self, train_state: train_state.TrainState, x0: jnp.ndarray, sde: SDE, solver: SDESolver, solve_keys: jnp.ndarray):
        # Create new solver instance with provided key
        solver = solver.from_sde(
            sde=sde,
            dt=solver.dt,
            total_time=solver.total_time,
            dim=x0.shape[2],
        )
        
        # Solve SDE for each sample with provided keys
        training_data, diffusion_history = jax.vmap(solver.solve, in_axes=(0, 0))(x0, solve_keys)
    
        
        # Process data - make sure xs and times have matching dimensions
        num_timesteps = training_data.shape[1]
        times = jnp.linspace(0, solver.total_time, num_timesteps)
        print(times.shape)
        xs = training_data
        
        # Compute Sigmas and drifts for all timesteps
        Sigmas = jax.vmap(jax.vmap(sde.Sigma, in_axes=(0, 0)), in_axes=(0, None))(xs, times)
        drifts = jax.vmap(jax.vmap(sde.drift_fn, in_axes=(0, 0)), in_axes=(0, None))(xs, times)

        def loss_fn(params):
            loss = Losses.ssm_dsm_loss(params, train_state, xs, times, x0, Sigmas, drifts, object_fn=self.object_fn)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss

    def train_epoch(self, train_state: train_state.TrainState, 
                   data_generator: DataGenerator, sde: SDE, solver: SDESolver, batch_size: int):
        x0 = self._generate_batch(data_generator, batch_size)
        solve_keys = self.key_monitor.split_keys(x0.shape[0])
        return self._train_step(train_state, x0, sde, solver, solve_keys)

    def train(self, train_state: train_state.TrainState, sde: SDE, solver: SDESolver, 
             data_generator: DataGenerator, epochs: int, batch_size: int):
        losses = jnp.zeros(epochs)
        # print the loss in the tqdm progress bar
        t = trange(epochs, desc="Bar desc")
        for i in t:
            train_state, loss = self.train_epoch(train_state, data_generator, sde, solver, batch_size)
            losses = losses.at[i].set(loss)
            t.set_description(f"Training loss: {loss}")
            t.refresh()
        return train_state, losses


class NeuralOpTrainer(Trainer):
    def __init__(self, seed: int = 0, landmark_num: int = 32):
        self.key_monitor = KeyMonitor(seed)
        self.object_fn = "Heng"
        self.landmark_num = landmark_num

    def train_state_init(self, model: nn.Module, lr: float = 1e-3, model_kwargs: dict = {}):
        """Initialize training state for neural operator model
        
        Args:
            model: Neural operator model (CTUNO1D or CTUNO2D)
            lr: Learning rate for optimizer
            model_kwargs: Dictionary containing:
                - x: Input data tensor
                - t: Time points tensor
                - object_fn: Optional loss function name
        """
        init_key = self.key_monitor.next_key()
        
        # Initialize model parameters - removed train parameter
        params = model.init(init_key, model_kwargs['x'], model_kwargs['t'])
        
        # Set object function if provided
        if 'object_fn' in model_kwargs:
            self.object_fn = model_kwargs['object_fn']
            
        # Initialize optimizer
        tx = optax.adam(lr)
        
        return train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx
        )

    def _generate_batch(self, data_generator: DataGenerator, landmark_num: int, batch_size: int):
        return data_generator.generate_data(landmark_num, batch_size)

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def _train_step(self, train_state: train_state.TrainState, x0: jnp.ndarray, sde: SDE, solver: SDESolver, solve_keys: jnp.ndarray):
        # Create new solver instance with provided key
        solver = solver.from_sde(
            sde=sde,
            dt=solver.dt,
            total_time=solver.total_time,
            dim=x0.shape[-1],
        )
        
        # Solve SDE for each sample with provided keys
        training_data, diffusion_history = jax.vmap(solver.solve, in_axes=(0, 0))(x0, solve_keys)
        
        # Process data
        num_timesteps = training_data.shape[1]
        times = jnp.linspace(0, solver.total_time, num_timesteps)
        xs = training_data
        
        # Compute Sigmas and drifts for all timesteps
        Sigmas = jax.vmap(jax.vmap(sde.Sigma, in_axes=(0, 0)), in_axes=(0, None))(xs, times)
        drifts = jax.vmap(jax.vmap(sde.drift_fn, in_axes=(0, 0)), in_axes=(0, None))(xs, times)

        def loss_fn(params):
            # Neural operator specific loss function
            loss = Losses.ssm_dsm_loss(params, train_state, xs, times, x0, Sigmas, drifts, object_fn=self.object_fn, with_x0=False)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss

    def train_epoch(self, train_state: train_state.TrainState, 
                   data_generator: DataGenerator, sde: SDE, solver: SDESolver, batch_size: int):
        x0 = self._generate_batch(data_generator, self.landmark_num, batch_size)
        solve_keys = self.key_monitor.split_keys(x0.shape[0])
        return self._train_step(train_state, x0, sde, solver, solve_keys)

    def train(self, train_state: train_state.TrainState, sde: SDE, solver: SDESolver, 
              data_generator: DataGenerator, epochs: int, batch_size: int):
        losses = jnp.zeros(epochs)
        # print the loss in the tqdm progress bar
        t = trange(epochs, desc="Training neural operator")
        for i in t:
            train_state, loss = self.train_epoch(train_state, data_generator, sde, solver, batch_size)
            losses = losses.at[i].set(loss)
            t.set_description(f"Training loss: {loss}")
            t.refresh()
        return train_state, losses
    