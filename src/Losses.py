import jax
import jax.numpy as jnp
from src.math.linalg import mat_weighted_norm

def ssm_dsm_loss(params, state, xs, times, x0, Sigmas, drifts, object_fn='Heng'):
    dt = times[1] - times[0]
    
    def single_step_loss(params, state, x_prev, x, t, x0, Sigma, Sigma_prev, drift_prev):
        pred_score = state.apply_fn(params, x, t, x0)
        
        if object_fn == 'Heng':
            # Add regularization as in notebook
            Sigma_prev = Sigma_prev + 1e-4 * jnp.eye(Sigma_prev.shape[0])
            Sigma_prev_inv = jnp.linalg.solve(Sigma_prev, jnp.eye(Sigma_prev.shape[0]))
            g_approx = -jnp.matmul(Sigma_prev_inv, (x - x_prev - dt * drift_prev))/dt
            diff = pred_score - g_approx
            loss = jnp.linalg.norm(jnp.matmul(diff.T, jnp.matmul(Sigma, diff))) ** 2
        else:
            # Novel version from notebook
            approx_stable = (x - x_prev - dt * drift_prev)
            loss = pred_score.T @ (Sigma * dt) @ pred_score + 2 * pred_score.T @ approx_stable
            loss = loss * dt
            
        return loss
    
    loss = jax.vmap(single_step_loss, in_axes=(None, None, 0, 0, 0, None, 0, 0, 0))(
        params, state, xs[:-1], xs[1:], times[:-1], x0, Sigmas[:-1], Sigmas[1:], drifts[:-1])
    return jnp.sum(loss, axis=0)
        