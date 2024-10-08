class EulerianSDE(BaseSDE):
    """ Eulerian SDE: dX(t) = Q^{1/2}(X(t)) dW(t),
        see ``Stochastic flows and shape bridges, S. Sommer et al.'' for details.
    """

    def __init__(self,
                 sigma: float = 1.0,
                 kappa: float = 0.1,
                 X0: FunctionData = None):
        super().__init__()
        self.sigma = sigma
        self.kappa = kappa
        self.X0 = X0

    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        # no drift term for this SDE
        return jnp.zeros_like(x)

    def g(self, t: float, x: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
        """ Diffusion term of the Eulerian SDE defined by the Gaussian kernel k(x, y) = sigma * exp(-||x-y||^2 / kappa^2).
            The covariance is computed between the landmarks.

        Args:
            t (float): time step.
            x (jnp.ndarray): flatten function evaluation x, shape (n_pts*co_dim, ).
            eps (float, optional): regularization to avoid singularity of the diffusion term. Defaults to 1e-4.

        Returns:
            jnp.ndarray: diffusion term, shape (n_pts*co_dim, n_pts*co_dim).
        """
        x = x.reshape(-1, 2)
        n_pts = x.shape[0]
        x = x + self.X0.sample(n_pts)
        kernel_fn = lambda x: self.sigma * jnp.exp(-jnp.sum(jnp.square(x), axis=-1) / self.kappa ** 2)
        dist = x[:, None, :] - x[None, :, :]
        kernel = kernel_fn(dist) + eps * jnp.eye(
            n_pts)  # Regularization to avoid singularity        Q_half = jnp.einsum("ij,kl->ikjl", kernel, jnp.eye(2))
        Q_half = Q_half.reshape(2 * n_pts, 2 * n_pts)
        return Q_half


class EulerianSDELandmarkIndependent(BaseSDE):
    """ Eulerian SDE: dX(t) = Q^{1/2}(X(t)) dW(t) with landmark-independent noise fields,
        see ``Stochastic flows and shape bridges, S. Sommer et al.'' for details.
    """

    def __init__(self,
                 sigma: float = 1.0,
                 kappa: float = 0.1,
                 X0: FunctionData = None,
                 grid_size: int = 50,
                 grid_range: Tuple[float, float] = (-0.5, 1.5)):
        super().__init__()
        self.sigma = sigma
        self.kappa = kappa
        self.X0 = X0
        self.grid_size = grid_size
        self.grid_range = grid_range

    @property
    def noise_dim(self):
        """ Noise dimension, set to be the number of points on the noise grid.
        """
        return self.grid_sz ** 2

    @property
    def grid(self):
        """ Noise grid.
        """
        grid = jnp.linspace(*self.grid_range, self.grid_sz)
        grid = jnp.stack(jnp.meshgrid(grid, grid, indexing="xy"), axis=-1)
        return grid

    def f(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x)

    def g(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        """ Diffusion term of the Eulerian SDE defined by the Gaussian kernel k(x, y) = sigma * exp(-||x-y||^2 / kappa^2).
            The covariance is computed between the landmarks and the grid points.

        Args:
            t (float): time step.
            x (jnp.ndarray): flatten function evaluation x, shape (n_pts*co_dim, ).
            eps (float, optional): regularization to avoid singularity of the diffusion term. Defaults to 1e-4.

        Returns:
            jnp.ndarray: diffusion term, shape (n_pts*co_dim, noise_dim*co_dim).
        """
        x = x.reshape(-1, 2)
        n_pts = x.shape[0]
        x = x + self.X0.sample(n_pts)
        kernel_fn = lambda x, y: self.sigma * jnp.exp(-0.5 * jnp.linalg.norm(x - y, axis=-1) ** 2 / self.kappa ** 2)
        Q_half = jax.vmap(
            jax.vmap(
                jax.vmap(
                    kernel_fn,
                    in_axes=(None, 0),
                    out_axes=0
                ),
                in_axes=(None, 1),
                out_axes=1
            ),
            in_axes=(0, None),
            out_axes=0
        )(x, self.grid)
        Q_half = Q_half.reshape(n_pts, self.noise_dim)
        Q_half = jnp.einsum("ij,kl->ikjl", Q_half, jnp.eye(2))
        Q_half = Q_half.reshape(2 * n_pts, 2 * self.noise_dim)
        return Q_half

    # def a(self, t: float, x: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
    #     g = self.g(t, x)
    #     return jnp.dot(g, g.T) + eps * jnp.eye(g.shape[0])