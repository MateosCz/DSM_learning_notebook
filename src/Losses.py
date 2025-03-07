import jax
import jax.numpy as jnp
from src.math.linalg import mat_weighted_norm

def ssm_dsm_loss(params, state, xs, times, x0, Sigmas, drifts, object_fn='Heng', with_x0=True):
    dt = times[1] - times[0]
    # dimensions:
    # Sigmas: (batch_size, num_timesteps, num_landmarks, dim)
    # xs: (batch_size, num_timesteps, num_landmarks, dim)
    # times: (num_timesteps,)
    # x0: (batch_size, num_landmarks, dim)
    # drifts: (batch_size, num_timesteps, num_landmarks, dim)

    # vmap over timesteps first then over batch size
    # the inner vmap is over the batch size
    loss = jax.vmap(batched_single_step_loss, in_axes=(None, #params
                                                        None, #state
                                                        1, #x_prev
                                                        1, #x
                                                        0, #t
                                                        None, #x0
                                                        1, #Sigma
                                                        1, #Sigma_prev
                                                        1, #drift_prev
                                                        None, #dt
                                                        None, #object_fn
                                                        None))(params, 
                                                               state, 
                                                               xs[:, :-1, ...], 
                                                               xs[:, 1:, ...], 
                                                               times[:-1], 
                                                               x0, 
                                                               Sigmas[:, 1:, ...], 
                                                               Sigmas[:, :-1, ...], 
                                                               drifts[:, :-1, ...], 
                                                               dt, 
                                                               object_fn,
                                                               with_x0)
    
    print(loss.shape)
    if object_fn == 'Heng':
        loss = jnp.mean(loss, axis=1)
        loss = jnp.sum(loss) * dt/2
    elif object_fn == 'Novel':
        loss = jnp.sum(loss)/xs.shape[0]
        loss = jnp.sum(loss)/2

    return loss
        
def single_step_loss(params, state, x_prev, x, t, x0, Sigma, Sigma_prev, drift_prev, dt, object_fn='Heng', with_x0=True):
    
        
    if object_fn == 'Heng':
        if with_x0:
            pred_score = state.apply_fn(params, x, t, x0)
        else:
            pred_score = state.apply_fn(params, x, t)
        # Add regularization as in notebook
        Sigma_prev = Sigma_prev + 5e-4 * jnp.eye(Sigma_prev.shape[0])
        Sigma_prev_inv = jnp.linalg.solve(Sigma_prev, jnp.eye(Sigma_prev.shape[0]))
        # Sigma_prev_inv = jnp.linalg.pinv(Sigma_prev)
        g_approx = -jnp.matmul(Sigma_prev_inv, (x - x_prev - dt * drift_prev))/dt
        diff = pred_score - g_approx
        loss = jnp.linalg.norm(jnp.matmul(diff.T, jnp.matmul(Sigma * dt, diff))) ** 2
    elif object_fn == 'Novel':
        # Novel version from notebook
        pred_score = state.apply_fn(params, x_prev, t-dt, x0)
        approx_stable = (x - x_prev - dt * drift_prev)
        loss = pred_score.T @ (Sigma_prev * dt) @ pred_score + 2 * pred_score.T @ approx_stable
        loss = loss * dt 
    return loss
# vmap over batch size, one batch's loss is mean at each timestep's loss
def batched_single_step_loss(params, state, x_prev, x, t, x0, Sigma, Sigma_prev, drift_prev, dt, object_fn='Heng', with_x0=True):
    batched_loss = jax.vmap(single_step_loss, in_axes=(None, #params
                                                        None, #state
                                                        0, #x_prev
                                                        0, #x
                                                        None, #t
                                                        0, #x0
                                                        0, #Sigma
                                                        0, #Sigma_prev
                                                        0, #drift_prev
                                                        None, #dt
                                                        None, #object_fn
                                                        None))(params, state, x_prev, x, t, x0, Sigma, Sigma_prev, drift_prev, dt, object_fn, with_x0)
    return batched_loss


# import jax
# import jax.numpy as jnp

# def ssm_dsm_loss(params, state, xs, times, x0, Sigmas, drifts, object_fn='Heng'):

#     dt = times[1] - times[0]

#     def single_step_loss(x, Sigma, grad):
#         pred_score = state.apply_fn(params, x, None, x0)  #score function
#         error = pred_score + grad 
#         return jnp.einsum("bi,bij,bj->b", error, Sigma, error)


#     gradss = (xs[:, 1:] - xs[:, :-1] - dt * drifts[:, :-1]) / dt

#     loss = jax.vmap(lambda x, Sigma, grad: single_step_loss(x, Sigma, grad),
#                     in_axes=(1, 1, 1))(xs[:, 1:], Sigmas[:, 1:], gradss)


#     loss = 0.5 * dt * jnp.mean(jnp.sum(loss, axis=1))
#     return loss




                                                        