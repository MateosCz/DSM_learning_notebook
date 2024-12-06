import jax
import jax.numpy as jnp

@jax.jit
def mat_weighted_norm(target, weight_mat):
    '''
    Compute the weighted norm of a matrix.
    args:
        target: the matrix to be normed
        weight_mat: the weight matrix
    return:
        the weighted norm of the matrix
    '''
    return jnp.matmul(target.T, jnp.matmul(weight_mat, target))