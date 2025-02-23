import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from typing import Callable, List
import jax.scipy as jscipy
from tqdm.autonotebook import tqdm
import flax.linen as nn
from typing import Optional



def normal_initializer(in_channels: int):
    """
    A simple normal initializer scaled by the square root of the number of input channels.
    """
    def init(key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype=dtype) / jnp.sqrt(in_channels)
    return init

class SpectralConv1d(eqx.Module):
    real_weights: jax.Array
    imag_weights: jax.Array
    in_channels: int
    out_channels: int
    modes: int
    out_grid_sz: int # output discretization size
    fft_norm: str = "forward"
    def __init__(
            self,
            in_channels,
            out_channels,
            modes,
            fft_norm,
            out_grid_sz,
            *,
            key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.modes = modes
        self.fft_norm = fft_norm
        self.out_grid_sz = out_grid_sz

        scale = 1.0 / (in_channels * out_channels * self.modes)

        real_key, imag_key = jax.random.split(key) # split the key for real and imaginary weights
        self.real_weights = jax.random.uniform(
            real_key,
            (modes//2+1, in_channels, out_channels),
            minval=-scale,
            maxval=+scale,
        ) # initialize the real weights
        self.imag_weights = jax.random.uniform(
            imag_key,
            (modes//2+1, in_channels, out_channels),
            minval=-scale,
            maxval=+scale,
        ) # initialize the imaginary weights

    def complex_mult1d(
            self,
            x_hat,
            w,
    ):
        return jnp.einsum("bij, ijk->bik", x_hat, w)
    
    def __call__(
            self,
            x,
    ):
        """ x shape: (batch, in_grid_sz, in_channels) 

            output shape: (batch, out_grid_sz, out_channels)
        """
        batch, in_grid_sz, in_channels = x.shape
        # shape of x_hat is (batch, in_grid_sz//2+1, in_channels)
        x_hat = jnp.fft.rfft(x, axis = -2, norm = self.fft_norm)
        # shape of x_hat_under_modes is (batch, self.modes//2+1, in_channels)
        x_hat_under_modes = x_hat[..., :self.modes//2+1, :]
        weights = self.real_weights + 1j * self.imag_weights
        # shape of out_hat_under_modes is (batch, self.modes//2+1, out_channels)
        out_hat_under_modes = self.complex_mult1d(x_hat_under_modes, weights)

        # shape of out_hat is (batch, self.in_grid_sz//2+1, out_channels)
        out_hat = jnp.zeros(
            (batch, self.in_grid_sz//2+1, self.out_channels),
            dtype=jnp.complex64
        )
        # set the data under the modes, truncate the data over the modes
        out_hat = out_hat.at[..., :self.modes//2+1, :].set(out_hat_under_modes)
        # shape of out is (batch, self.out_grid_sz, out_channels)
        out = jnp.fft.irfft(out_hat, n=self.out_grid_sz, axis = -2, norm = self.fft_norm)

        return out
    




class SpectralFreqTimeConv1D(eqx.Module):
    """
    Time-modulated integral kernel operator as proposed in the paper
    "Learning PDE Solution Operator for Continuous Modelling of Time-Series".

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        time_embedding_dim: Dimensionality of the time embedding.
        modes: Number of frequency modes.
        out_grid_sz: Output grid size. If None, output grid size equals input grid size.
        fft_norm: Normalization mode for FFT (e.g., 'forward').
    """
    # Hyperparameters
    in_channels: int
    out_channels: int
    time_embedding_dim: int
    modes: int
    out_grid_sz: Optional[int]
    fft_norm: str

    # Trainable parameters: real and imaginary parts of the spectral weights.
    weights_real: jnp.ndarray
    weights_imag: jnp.ndarray

    # Submodules: Two Dense layers (without bias) for transforming the time embedding.
    dense_real: eqx.nn.Linear
    dense_imag: eqx.nn.Linear

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        modes: int,
        out_grid_sz: Optional[int] = None,
        fft_norm: str = "forward",
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_dim = time_embedding_dim
        self.modes = modes
        self.out_grid_sz = out_grid_sz
        self.fft_norm = fft_norm

        # Split the random key for different initializations.
        key, key_w_real, key_w_imag, key_dense_real, key_dense_imag = jax.random.split(key, 5)
        weights_shape = (modes // 2 + 1, in_channels, out_channels)
        # Initialize weights using the normal_initializer.
        init_fn = normal_initializer(in_channels)
        self.weights_real = init_fn(key_w_real, weights_shape)
        self.weights_imag = init_fn(key_w_imag, weights_shape)

        # Initialize Dense layers for time embedding transformation (without bias).
        self.dense_real = eqx.nn.Linear(time_embedding_dim, modes // 2 + 1, key=key_dense_real, bias=False)
        self.dense_imag = eqx.nn.Linear(time_embedding_dim, modes // 2 + 1, key=key_dense_imag, bias=False)

    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the module.

        Args:
            x: Input tensor of shape (batch, in_grid_sz, in_channels).
            t_emb: Time embedding tensor of shape (batch, time_embedding_dim).

        Returns:
            Output tensor of shape (batch, out_grid_sz, out_channels).
        """
        batch, in_grid_sz, _ = x.shape
        # Determine the output grid size; if not specified, use the input grid size.
        out_grid_sz = in_grid_sz if self.out_grid_sz is None else self.out_grid_sz

        # Combine the real and imaginary parts to form complex weights.
        weights = self.weights_real + 1j * self.weights_imag

        # Compute the real FFT of the input along the spatial dimension (-2 axis).
        x_ft = jnp.fft.rfft(x, axis=-2, norm=self.fft_norm)

        # Initialize the frequency-domain output tensor with zeros.
        out_ft = jnp.zeros((batch, in_grid_sz // 2 + 1, self.out_channels), dtype=jnp.complex64)

        # Transform the time embedding using two Dense layers.
        t_emb_transf_real = self.dense_real(t_emb)
        t_emb_transf_imag = self.dense_imag(t_emb)
        t_emb_transf = t_emb_transf_real + 1j * t_emb_transf_imag

        # Time-modulate the weights.
        # Create an identity matrix of shape (modes//2+1, modes//2+1).
        eye = jnp.eye(self.modes // 2 + 1, dtype=t_emb_transf.dtype)
        # Multiply the time-transformed embedding with the identity to match the weight dimensions.
        # t_emb_transf[:, :, None] has shape (batch, modes//2+1, 1) and the result "modulated"
        # has shape (batch, modes//2+1, modes//2+1).
        modulated = t_emb_transf[:, :, None] * eye
        # Einstein summation to apply the modulation to the weights.
        #   modulated: (batch, modes//2+1, modes//2+1)
        #   weights: (modes//2+1, in_channels, out_channels)
        # Result: weights_modulated of shape (batch, modes//2+1, in_channels, out_channels)
        weights_modulated = jnp.einsum("bij,jkl->bikl", modulated, weights)

        # Truncate the Fourier transform of x to keep only the first modes//2+1 modes.
        x_ft_truncated = x_ft[:, : self.modes // 2 + 1, :]
        # Multiply the truncated Fourier transform with the modulated weights.
        #   x_ft_truncated: (batch, modes//2+1, in_channels)
        #   weights_modulated: (batch, modes//2+1, in_channels, out_channels)
        # The resulting tensor has shape (batch, modes//2+1, out_channels).
        out_ft_truncated = jnp.einsum("bij,bijk->bik", x_ft_truncated, weights_modulated)
        # Set the computed frequency components into the output tensor.
        out_ft = out_ft.at[:, : self.modes // 2 + 1, :].set(out_ft_truncated)

        # Perform the inverse FFT to convert back to the time domain.
        out = jnp.fft.irfft(out_ft, axis=-2, n=out_grid_sz, norm=self.fft_norm)
        return out






class SpectralConv2d(eqx.Module):
    """
    2D spectral convolution module for mapping functions u: R^2 -> R^(in_channels)
    to functions v: R^2 -> R^(out_channels).

    Attributes:
        real_weights1, imag_weights1: Real and imaginary parts of the first weight block,
            with shape (m, m, in_channels, out_channels), where m = modes//2 + 1.
        real_weights2, imag_weights2: Real and imaginary parts of the second weight block,
            with the same shape.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Total number of frequency modes (only the first modes//2+1 frequencies are used).
        out_grid_sz: Output grid size; if None, it defaults to the input grid size.
        fft_norm: Normalization mode for FFT (e.g., "forward").
    """
    real_weights1: jax.Array
    imag_weights1: jax.Array
    real_weights2: jax.Array
    imag_weights2: jax.Array
    in_channels: int
    out_channels: int
    modes: int
    out_grid_sz: Optional[int]  # If None, the output grid size equals the input grid size.
    fft_norm: str = "forward"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        fft_norm: str,
        out_grid_sz: Optional[int] = None,
        *,
        key
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.fft_norm = fft_norm
        self.out_grid_sz = out_grid_sz

        # Calculate m = modes//2 + 1 used for frequency slicing and weight tensor shape.
        m = modes // 2 + 1
        weight_shape = (m, m, in_channels, out_channels)
        init_fn = normal_initializer(in_channels)
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.real_weights1 = init_fn(key1, weight_shape)
        self.imag_weights1 = init_fn(key2, weight_shape)
        self.real_weights2 = init_fn(key3, weight_shape)
        self.imag_weights2 = init_fn(key4, weight_shape)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the spectral convolution.

        Args:
            x: Input tensor with shape (batch, in_grid_sz, in_grid_sz, in_channels).

        Returns:
            Output tensor with shape (batch, out_grid_sz, out_grid_sz, out_channels).
            If self.out_grid_sz is None, the output grid size equals the input grid size.
        """
        batch, in_grid_sz, _, _ = x.shape
        m = self.modes // 2 + 1
        # Use the input grid size as output grid size if not specified.
        out_grid_sz = in_grid_sz if self.out_grid_sz is None else self.out_grid_sz

        # Perform 2D real FFT on the input along axes (1, 2).
        # The resulting shape is (batch, in_grid_sz, in_grid_sz//2+1, in_channels).
        x_ft = jnp.fft.rfft2(x, axes=(1, 2), norm=self.fft_norm)

        # Initialize the output frequency-domain tensor with zeros.
        out_ft = jnp.zeros((batch, in_grid_sz, in_grid_sz//2+1, self.out_channels), dtype=jnp.complex64)

        # Combine real and imaginary parts for both weight blocks.
        weights1 = self.real_weights1 + 1j * self.imag_weights1
        weights2 = self.real_weights2 + 1j * self.imag_weights2

        # Apply convolution with the first weight block on the first m frequencies.
        x_ft1 = jnp.einsum("bijk,ijkl->bijl", x_ft[:, :m, :m, :], weights1)
        # Apply convolution with the second weight block on the last m frequencies along axis 1.
        x_ft2 = jnp.einsum("bijk,ijkl->bijl", x_ft[:, -m:, :m, :], weights2)

        # Insert the computed frequency components into the corresponding frequency bands.
        out_ft = out_ft.at[:, :m, :m, :].set(x_ft1)
        out_ft = out_ft.at[:, -m:, :m, :].set(x_ft2)

        # Perform the inverse 2D FFT to convert the frequency-domain result back to the spatial domain.
        out = jnp.fft.irfft2(out_ft, s=(out_grid_sz, out_grid_sz), axes=(1, 2), norm=self.fft_norm)
        return out

class SpectralFreqTimeConv2d(eqx.Module):
    """
    Time-modulated integral kernel operator for mapping functions 
    u: R^2 -> R^(in_channels) to functions v: R^2 -> R^(out_channels), with time modulation.
    This implementation follows the naming convention of SpectralConv2d.

    Attributes:
        real_weights1, imag_weights1: Real and imaginary parts of the first weight block,
            with shape (m, m, in_channels, out_channels), where m = modes//2 + 1.
        real_weights2, imag_weights2: Real and imaginary parts of the second weight block,
            with the same shape.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        t_emb_dim: Dimensionality of the time embedding.
        modes: Total number of frequency modes (only the first modes//2 + 1 frequencies are used).
        out_grid_sz: Output grid size; if None, it defaults to the input grid size.
        fft_norm: Normalization mode for FFT (e.g., "forward").
        dense_t1_real, dense_t1_imag: Dense layers (without bias) that transform the time embedding
            into modulation coefficients for the first weight block. Each outputs a tensor of shape (batch, m).
        dense_t2_real, dense_t2_imag: Dense layers for transforming the time embedding for the second weight block.
    """
    real_weights1: jax.Array
    imag_weights1: jax.Array
    real_weights2: jax.Array
    imag_weights2: jax.Array
    in_channels: int
    out_channels: int
    t_emb_dim: int
    modes: int
    out_grid_sz: Optional[int]
    fft_norm: str = "forward"

    dense_t1_real: eqx.nn.Linear
    dense_t1_imag: eqx.nn.Linear
    dense_t2_real: eqx.nn.Linear
    dense_t2_imag: eqx.nn.Linear

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_emb_dim: int,
        modes: int,
        fft_norm: str,
        out_grid_sz: Optional[int] = None,
        *,
        key
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_emb_dim = t_emb_dim
        self.modes = modes
        self.fft_norm = fft_norm
        self.out_grid_sz = out_grid_sz

        # Compute m = modes//2 + 1 for frequency slicing and weight shapes.
        m = modes // 2 + 1
        weight_shape = (m, m, in_channels, out_channels)
        init_fn = normal_initializer(in_channels)

        # Split keys for weights and dense layers.
        key, key_rw1, key_iw1, key_rw2, key_iw2, key_dense1, key_dense2, key_dense3, key_dense4 = jax.random.split(key, 9)
        
        # Initialize real and imaginary parts of the first weight block.
        self.real_weights1 = init_fn(key_rw1, weight_shape)
        self.imag_weights1 = init_fn(key_iw1, weight_shape)
        
        # Initialize real and imaginary parts of the second weight block.
        self.real_weights2 = init_fn(key_rw2, weight_shape)
        self.imag_weights2 = init_fn(key_iw2, weight_shape)

        # Initialize Dense layers for time embedding transformation (without bias).
        # Each layer maps from t_emb_dim to m (i.e., modes//2 + 1).
        self.dense_t1_real = eqx.nn.Linear(t_emb_dim, m, use_bias=False, key=key_dense1)
        self.dense_t1_imag = eqx.nn.Linear(t_emb_dim, m, use_bias=False, key=key_dense2)
        self.dense_t2_real = eqx.nn.Linear(t_emb_dim, m, use_bias=False, key=key_dense3)
        self.dense_t2_imag = eqx.nn.Linear(t_emb_dim, m, use_bias=False, key=key_dense4)

    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the time-modulated spectral convolution.

        Args:
            x: Input tensor with shape (batch, in_grid_sz, in_grid_sz, in_channels).
            t_emb: Time embedding tensor with shape (batch, t_emb_dim).

        Returns:
            Output tensor with shape (batch, out_grid_sz, out_grid_sz, out_channels).
            If out_grid_sz is None, the output grid size equals the input grid size.
        """
        batch, in_grid_sz, _, _ = x.shape
        m = self.modes // 2 + 1
        out_grid_sz = in_grid_sz if self.out_grid_sz is None else self.out_grid_sz

        # 1. Perform 2D real FFT on the input.
        #    x_ft shape: (b, in_grid_sz, in_grid_sz//2 + 1, in_channels)
        x_ft = jnp.fft.rfft2(x, axes=(1, 2), norm=self.fft_norm)

        # 2. Transform time embedding into modulation coefficients.
        #    Each dense layer outputs a tensor of shape (b, m).
        t_emb_transf1 = self.dense_t1_real(t_emb) + 1j * self.dense_t1_imag(t_emb)  # (b, m)
        t_emb_transf2 = self.dense_t2_real(t_emb) + 1j * self.dense_t2_imag(t_emb)  # (b, m)

        # 3. Create an identity matrix and compute modulation matrices.
        #    eye_m shape: (m, m)
        eye_m = jnp.eye(m, dtype=t_emb_transf1.dtype)
        #    modulation1 and modulation2 shapes: (b, m, m)
        modulation1 = t_emb_transf1[:, :, None] * eye_m
        modulation2 = t_emb_transf2[:, :, None] * eye_m

        # 4. Combine weights into complex tensors.
        #    weights_complex shape: (m, m, in_channels, out_channels)
        weights1_complex = self.real_weights1 + 1j * self.imag_weights1
        weights2_complex = self.real_weights2 + 1j * self.imag_weights2

        # 5. Modulate the weights with the time embedding.
        #    After modulation, weights become sample-specific.
        #    weights1_modulated and weights2_modulated shapes: (b, m, m, in_channels, out_channels)
        weights1_modulated = jnp.einsum("bij,jklm->biklm", modulation1, weights1_complex)
        weights2_modulated = jnp.einsum("bij,jklm->biklm", modulation2, weights2_complex)

        # 6. Initialize the output frequency-domain tensor.
        #    out_ft shape: (b, in_grid_sz, in_grid_sz//2 + 1, out_channels)
        out_ft = jnp.zeros((batch, in_grid_sz, in_grid_sz//2 + 1, self.out_channels), dtype=jnp.complex64)

        # 7. Apply the modulated weights to the selected frequency components.
        #    For the first weight block:
        #      x_ft[:, :m, :m, :] has shape (b, m, m, in_channels)
        #      Result x_ft1 shape: (b, m, m, out_channels)
        x_ft1 = jnp.einsum("bijk,bijkl->bijl", x_ft[:, :m, :m, :], weights1_modulated)
        #    For the second weight block:
        #      x_ft[:, -m:, :m, :] has shape (b, m, m, in_channels)
        #      Result x_ft2 shape: (b, m, m, out_channels)
        x_ft2 = jnp.einsum("bijk,bijkl->bijl", x_ft[:, -m:, :m, :], weights2_modulated)

        # 8. Insert the computed frequency components into the output frequency tensor.
        out_ft = out_ft.at[:, :m, :m, :].set(x_ft1)
        out_ft = out_ft.at[:, -m:, :m, :].set(x_ft2)

        # 9. Perform the inverse 2D FFT to convert back to the spatial domain.
        #    Output shape: (b, out_grid_sz, out_grid_sz, out_channels)
        out = jnp.fft.irfft2(out_ft, s=(out_grid_sz, out_grid_sz), axes=(1, 2), norm=self.fft_norm)
        return out