import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import abc
from jax.lax import scan


class Trainer(abc.ABC):
    model: nn.Module
    @abc.abstractmethod
    def train_state_init(self, key: jnp.ndarray, lr: float = 1e-3, model_kwargs: dict = {}):
        pass

    @abc.abstractmethod
    def train_step(self, train_state: train_state.TrainState, model: nn.Module, model_inputs: dict):
        pass

class DsmTrainer(Trainer):
    def train_state_init(self, key: jnp.ndarray, lr: float = 1e-3, model_kwargs: dict = {}):
        params = self.model.init(key, model_kwargs['x'], model_kwargs['t'], model_kwargs['x0'])
        tx = optax.adam(lr)
        return train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

    def train_step(self, train_state: train_state.TrainState, model: nn.Module, model_inputs: dict):
        def loss_fn(params):
            pred = model.apply(params, model_inputs['x'], model_inputs['t'], model_inputs['x0'])
            return jnp.mean(jnp.square(pred - model_inputs['y']))
        grads = jax.grad(loss_fn)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, None

    