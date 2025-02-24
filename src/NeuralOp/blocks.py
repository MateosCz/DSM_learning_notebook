from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn

def get_activation_fn(activation_str):
    if activation_str.lower() == 'relu':
        return nn.relu
    elif activation_str.lower() == 'tanh':
        return nn.tanh
    elif activation_str.lower() == 'silu':
        return nn.silu
    elif activation_str.lower() == 'gelu':
        return nn.gelu
    elif activation_str.lower() == 'leaky_relu':
        return nn.leaky_relu
    elif activation_str.lower() == 'elu':
        return nn.elu
    else:
        raise ValueError(f"Unknown activation function: {activation_str}")

def normal_initializer(input_co_dim: int):
    return nn.initializers.normal(stddev=jnp.sqrt(1.0/(2.0*input_co_dim)))

### Fourier Layers ###
    
class SpectralConv1D(nn.Module):
    """ Integral kernel operator for mapping functions (u: R -> R^{in_co_dim}) to functions (v: R -> R^{out_co_dim}) """
    in_co_dim: int
    out_co_dim: int
    n_modes: int
    out_grid_sz: int = None
    fft_norm: str = "forward"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (in_grid_sz, in_co_dim) 
            output shape: (out_grid_sz, out_co_dim)
        """
        in_grid_sz = x.shape[0]
        out_grid_sz = in_grid_sz if self.out_grid_sz is None else self.out_grid_sz
        weights_shape = (self.n_modes//2+1, self.in_co_dim, self.out_co_dim)
        weights_real = self.param(
            'weights(real)',
            normal_initializer(self.in_co_dim),
            weights_shape
        )
        weights_imag = self.param(
            'weights(imag)',
            normal_initializer(self.in_co_dim),
            weights_shape
        )
        weights = weights_real + 1j*weights_imag

        x_ft = jnp.fft.rfft(x, axis=0, norm=self.fft_norm)

        out_ft = jnp.zeros((in_grid_sz//2+1, self.out_co_dim), dtype=jnp.complex64)
        x_ft = jnp.einsum("ij,ijk->ik", x_ft[:self.n_modes//2+1, :], weights)
        out_ft = out_ft.at[:self.n_modes//2+1, :].set(x_ft)

        out = jnp.fft.irfft(out_ft, axis=0, n=out_grid_sz, norm=self.fft_norm)
        return out

class SpectralConv2D(nn.Module):
    """ Integral kernel operator for mapping functions (u: R^2 -> R^{in_co_dim}) to functions (v: R^2 -> R^{out_co_dim}) """
    in_co_dim: int
    out_co_dim: int
    n_modes: int
    out_grid_sz: int = None
    fft_norm: str = "forward"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (in_grid_sz, in_grid_sz, in_co_dim) 
            output shape: (out_grid_sz, out_grid_sz, out_co_dim)
        """
        in_grid_sz = x.shape[0]
        out_grid_sz = in_grid_sz if self.out_grid_sz is None else self.out_grid_sz
        weights_shape = (self.n_modes//2+1, self.n_modes//2+1, self.in_co_dim, self.out_co_dim)
        weights1_real = self.param(
            'weights1(real)',
            normal_initializer(self.in_co_dim),
            weights_shape
        )
        weights1_imag = self.param(
            'weights1(imag)',
            normal_initializer(self.in_co_dim),
            weights_shape
        )
        weights1 = weights1_real + 1j*weights1_imag
        weights2_real = self.param(
            'weights2(real)',
            normal_initializer(self.in_co_dim),
            weights_shape
        )
        weights2_imag = self.param(
            'weights2(imag)',
            normal_initializer(self.in_co_dim),
            weights_shape
        )
        weights2 = weights2_real + 1j*weights2_imag

        x_ft = jnp.fft.rfft2(x, axes=(0, 1), norm=self.fft_norm)

        out_ft = jnp.zeros((in_grid_sz, in_grid_sz//2+1, self.out_co_dim), dtype=jnp.complex64)
        x_ft1 = jnp.einsum("ijk,ijkl->ijl", x_ft[:self.n_modes//2+1, :self.n_modes//2+1, :], weights1)
        x_ft2 = jnp.einsum("ijk,ijkl->ijl", x_ft[-(self.n_modes//2+1):, :self.n_modes//2+1, :], weights2)
        out_ft = out_ft.at[:self.n_modes//2+1, :self.n_modes//2+1, :].set(x_ft1)
        out_ft = out_ft.at[-(self.n_modes//2+1):, :self.n_modes//2+1, :].set(x_ft2)

        out = jnp.fft.irfft2(out_ft, s=(out_grid_sz, out_grid_sz), axes=(0, 1), norm=self.fft_norm)
        return out
    
    
class SpectralFreqTimeConv1D(nn.Module):
    """ Time modulated integral kernel operator """
    in_co_dim: int
    out_co_dim: int
    t_emb_dim: int
    n_modes: int
    out_grid_sz: int = None
    fft_norm: str = "forward"

    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (in_grid_sz, in_co_dim),
            t_emb shape: (t_emb_dim)

            output shape: (out_grid_sz, out_co_dim)
        """
        in_grid_sz = x.shape[0]
        out_grid_sz = in_grid_sz if self.out_grid_sz is None else self.out_grid_sz
        weights_shape = (self.n_modes//2+1, self.in_co_dim, self.out_co_dim)
        weights_real = self.param(
            'weights(real)',
            normal_initializer(self.in_co_dim),
            weights_shape,
        )
        weights_imag = self.param(
            'weights(imag)',
            normal_initializer(self.in_co_dim),
            weights_shape,
        )
        weights = weights_real + 1j*weights_imag

        x_ft = jnp.fft.rfft(x, axis=0, norm=self.fft_norm)

        out_ft = jnp.zeros((in_grid_sz//2+1, self.out_co_dim), dtype=jnp.complex64)

        t_emb_transf_real = nn.Dense(
            self.n_modes//2+1,
            use_bias=False,
        )(t_emb)
        t_emb_transf_imag = nn.Dense(
            self.n_modes//2+1,
            use_bias=False,
        )(t_emb)
        t_emb_transf = t_emb_transf_real + 1j*t_emb_transf_imag

        weights = jnp.einsum("i,ijk->ijk", t_emb_transf[:self.n_modes//2+1], weights)
        x_ft = jnp.einsum("ij,ijk->ik", x_ft[:self.n_modes//2+1, :], weights)
        out_ft = out_ft.at[:self.n_modes//2+1, :].set(x_ft)

        out = jnp.fft.irfft(out_ft, axis=0, n=out_grid_sz, norm=self.fft_norm)
        return out

class SpectralFreqTimeConv2D(nn.Module):
    """ Time modulated integral kernel operator """
    in_co_dim: int
    out_co_dim: int
    t_emb_dim: int
    n_modes: int
    out_grid_sz: int = None
    fft_norm: str = "forward"

    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (in_grid_sz, in_grid_sz, in_co_dim),
            t_emb shape: (t_emb_dim)

            output shape: (out_grid_sz, out_grid_sz, out_co_dim)
        """
        in_grid_sz = x.shape[0]
        out_grid_sz = in_grid_sz if self.out_grid_sz is None else self.out_grid_sz
        weights_shape = (self.n_modes//2+1, self.n_modes//2+1, self.in_co_dim, self.out_co_dim)
        weights1_real = self.param(
            'weights1(real)',
            normal_initializer(self.in_co_dim),
            weights_shape,
        )
        weights1_imag = self.param(
            'weights1(imag)',
            normal_initializer(self.in_co_dim),
            weights_shape,
        )
        weights1 = weights1_real + 1j*weights1_imag
        weights2_real = self.param(
            'weights2(real)',
            normal_initializer(self.in_co_dim),
            weights_shape,
        )
        weights2_imag = self.param(
            'weights2(imag)',
            normal_initializer(self.in_co_dim),
            weights_shape,
        )
        weights2 = weights2_real + 1j*weights2_imag

        x_ft = jnp.fft.rfft2(x, axes=(0, 1), norm=self.fft_norm)

        out_ft = jnp.zeros((in_grid_sz, in_grid_sz//2+1, self.out_co_dim), dtype=jnp.complex64)

        t_emb_transf1_real = nn.Dense(
            self.n_modes//2+1,
            use_bias=False,
        )(t_emb)        
        t_emb_transf1_imag = nn.Dense(
            self.n_modes//2+1,
            use_bias=False,
        )(t_emb)
        t_emb_transf1 = t_emb_transf1_real + 1j*t_emb_transf1_imag
        t_emb_transf2_real = nn.Dense(
            self.n_modes//2+1,
            use_bias=False,
        )(t_emb)
        t_emb_transf2_imag = nn.Dense(
            self.n_modes//2+1,
            use_bias=False,
        )(t_emb)
        t_emb_transf2 = t_emb_transf2_real + 1j*t_emb_transf2_imag

        weights1 = jnp.einsum("i,ijkl->ijkl", t_emb_transf1[:self.n_modes//2+1], weights1)
        weights2 = jnp.einsum("i,ijkl->ijkl", t_emb_transf2[:self.n_modes//2+1], weights2)

        x_ft1 = jnp.einsum('ijk,ijkl->ijl', x_ft[:self.n_modes//2+1, :self.n_modes//2+1, :], weights1)
        x_ft2 = jnp.einsum('ijk,ijkl->ijl', x_ft[-(self.n_modes//2+1):, :self.n_modes//2+1, :], weights2)
        out_ft = out_ft.at[:self.n_modes//2+1, :self.n_modes//2+1, :].set(x_ft1)
        out_ft = out_ft.at[-(self.n_modes//2+1):, :self.n_modes//2+1, :].set(x_ft2)

        out = jnp.fft.irfft2(out_ft, s=(out_grid_sz, out_grid_sz), axes=(0, 1), norm=self.fft_norm)
        return out
    
class TimeConv1D(nn.Module):
    out_co_dim: int
    out_grid_sz: int = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (in_grid_sz, in_co_dim),
            t_emb shape: (t_emb_dim)
            
            output shape: (out_grid_sz, out_co_dim)
        """
        x = nn.Conv(features=self.out_co_dim, kernel_size=(1,), padding="VALID")(x)
        weights = self.param(
            'weights',
            nn.initializers.normal(),
            (self.out_co_dim, self.out_co_dim)
        )
        psi_t = nn.Dense(
            2 * self.out_co_dim,
            use_bias=False
        )(t_emb)
        w_t, b_t = jnp.split(psi_t, 2, axis=-1)
        x = jnp.einsum("ij,j,lk->li", weights, w_t, x)
        x = x + b_t
        if self.out_grid_sz is not None:
            x = jax.image.resize(x, (self.out_grid_sz, self.out_co_dim), method="nearest")
        return x
    
class TimeConv2D(nn.Module):
    out_co_dim: int
    out_grid_sz: int = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (in_grid_sz, in_grid_sz, in_co_dim),
            t_emb shape: (t_emb_dim)
            
            output shape: (out_grid_sz, out_grid_sz, out_co_dim)
        """
        x = nn.Conv(features=self.out_co_dim, kernel_size=(1, 1), padding="VALID")(x)
        weights = self.param(
            'weights',
            nn.initializers.normal(),
            (self.out_co_dim, self.out_co_dim)
        )
        psi_t = nn.Dense(
            2 * self.out_co_dim,
            use_bias=False
        )(t_emb)
        w_t, b_t = jnp.split(psi_t, 2, axis=-1)
        x = jnp.einsum("ij,j,lmk->lmi", weights, w_t, x)
        x = x + b_t[None, None, :]
        if self.out_grid_sz is not None:
            x = jax.image.resize(x, (self.out_grid_sz, self.out_grid_sz, self.out_co_dim), method="nearest")
        return x

### FNO Blocks ###
class CTUNOBlock1D(nn.Module):
    in_co_dim: int
    out_co_dim: int
    t_emb_dim: int
    n_modes: int
    out_grid_sz: int = None
    fft_norm: str = "forward"
    norm: str = "instance"
    act: str = "relu"
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (in_grid_sz, in_co_dim),
            t_emb shape: (t_emb_dim)

            output shape: (out_grid_sz, out_co_dim)
        """
        x_spec_out = SpectralFreqTimeConv1D(
            self.in_co_dim,
            self.out_co_dim,
            self.t_emb_dim,
            self.n_modes,
            self.out_grid_sz,
            self.fft_norm
        )(x, t_emb)
        x_res_out = TimeConv1D(
            self.out_co_dim,
            self.out_grid_sz,
        )(x, t_emb)
        x_out = x_spec_out + x_res_out
        # if self.norm.lower() == "instance":
        #     x_out = nn.InstanceNorm()(x_out)

        return get_activation_fn(self.act)(x_out)
    
class CTUNOBlock2D(nn.Module):
    in_co_dim: int
    out_co_dim: int
    t_emb_dim: int
    n_modes: int
    out_grid_sz: int = None
    fft_norm: str = "forward"
    norm: str = "instance"
    act: str = "relu"
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (in_grid_sz, in_grid_sz, in_co_dim),
            t_emb shape: (t_emb_dim)

            output shape: (out_grid_sz, out_grid_sz, out_co_dim)
        """
        x_spec_out = SpectralFreqTimeConv2D(
            self.in_co_dim,
            self.out_co_dim,
            self.t_emb_dim,
            self.n_modes,
            self.out_grid_sz,
            self.fft_norm
        )(x, t_emb)
        x_res_out = TimeConv2D(
            self.out_co_dim,
            self.out_grid_sz,
        )(x, t_emb)
        x_out = x_spec_out + x_res_out
        # if self.norm.lower() == "instance":
        #     x_out = nn.InstanceNorm()(x_out)

        return get_activation_fn(self.act)(x_out)
    
    
class TimeEmbedding(nn.Module):
    """ Sinusoidal time step embedding """
    t_emb_dim: int
    scaling: float = 100.0
    max_period: float = 10000.0

    @nn.compact
    def __call__(self, t):
        """ t shape: (,) """
        pe = jnp.empty((self.t_emb_dim,))
        factor = self.scaling * t * jnp.exp(jnp.arange(0, self.t_emb_dim, 2) * -(jnp.log(self.max_period) / self.t_emb_dim))
        pe = pe.at[0::2].set(jnp.sin(factor))
        pe = pe.at[1::2].set(jnp.cos(factor))
        return pe