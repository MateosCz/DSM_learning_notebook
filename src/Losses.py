import jax
import jax.numpy as jnp
from src.math.linalg import mat_weighted_norm
import functools
# @jax.jit
# def dsm_loss(params, state, data, key):


def ssm_dsm_loss(params, state, xs, times, x0, Sigmas, drifts):
    dt = times[1] - times[0]
    def single_step_loss(params, state, x_prev, x, t, x0, Sigma, Sigma_prev, drift_prev):
        pred_score = state.apply_fn(params, x, t, x0)
        Sigma_prev_inv = jnp.linalg.solve(Sigma_prev, jnp.eye(Sigma_prev.shape[0]))
        g_approx = -jnp.matmul(Sigma_prev_inv, (x - x_prev - dt * drift_prev))/dt
        diff = pred_score - g_approx
        normed_diff = mat_weighted_norm(diff, Sigma)
        expected_loss = jnp.mean(normed_diff, axis=0) # the expected loss is the expectation on x0
        return expected_loss
    loss = jax.vmap(single_step_loss, in_axes=(None, None, 0, 0, None, None, 0, 0,0))(params, state, xs[:-1], xs[1:], times[:-1], x0, Sigmas[:-1], Sigmas[1:], drifts[:-1])
    loss = jnp.sum(loss)
    loss = loss * dt /2
    return loss
        
