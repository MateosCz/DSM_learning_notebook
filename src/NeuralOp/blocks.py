from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Callable, List
from src.math.geometry import batch_radius_neighbors, find_nearest_neighbors_in_ball
from dataclasses import field
import numpy as np
from dataclasses import field
from typing import Optional, List, Tuple, Union, Any
from src.math.sparse import segment_csr, CSRMatrix
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

# class SpectralConv2D(nn.Module):
#     """ Integral kernel operator for mapping functions (u: R^2 -> R^{in_co_dim}) to functions (v: R^2 -> R^{out_co_dim}) """
#     in_co_dim: int
#     out_co_dim: int
#     n_modes: int
#     out_grid_sz: int = None
#     fft_norm: str = "forward"

#     @nn.compact
#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         """ x shape: (in_grid_sz, in_grid_sz, in_co_dim) 
#             output shape: (out_grid_sz, out_grid_sz, out_co_dim)
#         """
#         in_grid_sz = x.shape[0]
#         out_grid_sz = in_grid_sz if self.out_grid_sz is None else self.out_grid_sz
#         weights_shape = (self.n_modes//2+1, self.n_modes//2+1, self.in_co_dim, self.out_co_dim)
#         weights1_real = self.param(
#             'weights1(real)',
#             normal_initializer(self.in_co_dim),
#             weights_shape
#         )
#         weights1_imag = self.param(
#             'weights1(imag)',
#             normal_initializer(self.in_co_dim),
#             weights_shape
#         )
#         weights1 = weights1_real + 1j*weights1_imag
#         weights2_real = self.param(
#             'weights2(real)',
#             normal_initializer(self.in_co_dim),
#             weights_shape
#         )
#         weights2_imag = self.param(
#             'weights2(imag)',
#             normal_initializer(self.in_co_dim),
#             weights_shape
#         )
#         weights2 = weights2_real + 1j*weights2_imag

#         x_ft = jnp.fft.rfft2(x, axes=(0, 1), norm=self.fft_norm)

#         out_ft = jnp.zeros((in_grid_sz, in_grid_sz//2+1, self.out_co_dim), dtype=jnp.complex64)
#         x_ft1 = jnp.einsum("ijk,ijkl->ijl", x_ft[:self.n_modes//2+1, :self.n_modes//2+1, :], weights1)
#         x_ft2 = jnp.einsum("ijk,ijkl->ijl", x_ft[-(self.n_modes//2+1):, :self.n_modes//2+1, :], weights2)
#         out_ft = out_ft.at[:self.n_modes//2+1, :self.n_modes//2+1, :].set(x_ft1)
#         out_ft = out_ft.at[-(self.n_modes//2+1):, :self.n_modes//2+1, :].set(x_ft2)

#         out = jnp.fft.irfft2(out_ft, s=(out_grid_sz, out_grid_sz), axes=(0, 1), norm=self.fft_norm)
#         return out
    

class SpectralConv2D(nn.Module):
    """ Integral kernel operator for mapping functions (u: R^2 -> R^{in_co_dim}) to functions (v: R^2 -> R^{out_co_dim}) """
    in_co_dim: int
    out_co_dim: int
    n_modes: int  # Number of Fourier modes to use in each spatial dimension
    out_grid_sz: int = None
    fft_norm: str = "forward"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (in_grid_sz, in_grid_sz, in_co_dim) 
            output shape: (out_grid_sz, out_grid_sz, out_co_dim)
        """
        in_grid_sz = x.shape[0]
        out_grid_sz = in_grid_sz if self.out_grid_sz is None else self.out_grid_sz

        # 仅保留正频率部分的权重
        weights_shape = (self.n_modes//2 + 1, self.n_modes//2 + 1, self.in_co_dim, self.out_co_dim)
        weights_real = self.param(
            'weights_real',
            nn.initializers.normal(stddev=1/(self.in_co_dim + self.out_co_dim)),
            weights_shape
        )
        weights_imag = self.param(
            'weights_imag',
            nn.initializers.normal(stddev=1/(self.in_co_dim + self.out_co_dim)),
            weights_shape
        )
        weights = weights_real + 1j * weights_imag

        x_ft = jnp.fft.rfft2(x, axes=(0, 1), norm=self.fft_norm)  # Shape: (in_grid_sz, in_grid_sz//2 + 1, in_co_dim)

        # 仅处理正频率的低频部分
        x_ft_trunc = x_ft[:self.n_modes//2 + 1, :self.n_modes//2 + 1, :]
        out_ft_upper = jnp.einsum("ijk,ijkl->ijl", x_ft_trunc, weights)

        # 初始化输出频谱并填充处理后的正频率部分
        out_ft = jnp.zeros((in_grid_sz, in_grid_sz//2 + 1, self.out_co_dim), dtype=jnp.complex64)
        out_ft = out_ft.at[:self.n_modes//2 + 1, :self.n_modes//2 + 1, :].set(out_ft_upper)

        # 自动填充轴0的负频率共轭部分（保持对称性）
        # 注意：对于实数信号，x_ft的轴0索引i和in_grid_sz -i是共轭对称的
        # 这里仅处理正频率，负频率由共轭对称性自动生成
        # 因此无需额外处理负频率区域

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

        # Fix: Ensure dimensions match before einsum by taking the minimum size
        modes_to_use = min(self.n_modes//2+1, x_ft.shape[0], weights.shape[0])
        
        # Apply time modulation to weights with consistent size
        weights_slice = weights[:modes_to_use]
        t_emb_slice = t_emb_transf[:modes_to_use]
        modulated_weights = jnp.einsum("i,ijk->ijk", t_emb_slice, weights_slice)
        
        # Apply modulated weights to frequency components
        x_ft_slice = x_ft[:modes_to_use, :]
        result = jnp.einsum("ij,ijk->ik", x_ft_slice, modulated_weights)
        
        # Update output at corresponding frequencies
        out_ft = out_ft.at[:modes_to_use, :].set(result)

        out = jnp.fft.irfft(out_ft, axis=0, n=out_grid_sz, norm=self.fft_norm)
        return out

# class SpectralFreqTimeConv2D(nn.Module):
#     """ Time modulated integral kernel operator """
#     in_co_dim: int
#     out_co_dim: int
#     t_emb_dim: int
#     n_modes: int
#     out_grid_sz: int = None
#     fft_norm: str = "forward"

#     @nn.compact
#     def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray) -> jnp.ndarray:
#         """ x shape: (in_grid_sz, in_grid_sz, in_co_dim),
#             t_emb shape: (t_emb_dim)

#             output shape: (out_grid_sz, out_grid_sz, out_co_dim)
#         """
#         in_grid_sz = x.shape[0]
#         out_grid_sz = in_grid_sz if self.out_grid_sz is None else self.out_grid_sz
#         weights_shape = (self.n_modes//2+1, self.n_modes//2+1, self.in_co_dim, self.out_co_dim)
#         weights1_real = self.param(
#             'weights1(real)',
#             normal_initializer(self.in_co_dim),
#             weights_shape,
#         )
#         weights1_imag = self.param(
#             'weights1(imag)',
#             normal_initializer(self.in_co_dim),
#             weights_shape,
#         )
#         weights1 = weights1_real + 1j*weights1_imag
#         weights2_real = self.param(
#             'weights2(real)',
#             normal_initializer(self.in_co_dim),
#             weights_shape,
#         )
#         weights2_imag = self.param(
#             'weights2(imag)',
#             normal_initializer(self.in_co_dim),
#             weights_shape,
#         )
#         weights2 = weights2_real + 1j*weights2_imag

#         x_ft = jnp.fft.rfft2(x, axes=(0, 1), norm=self.fft_norm)

#         out_ft = jnp.zeros((in_grid_sz, in_grid_sz//2+1, self.out_co_dim), dtype=jnp.complex64)

#         t_emb_transf1_real = nn.Dense(
#             self.n_modes//2+1,
#             use_bias=False,
#         )(t_emb)        
#         t_emb_transf1_imag = nn.Dense(
#             self.n_modes//2+1,
#             use_bias=False,
#         )(t_emb)
#         t_emb_transf1 = t_emb_transf1_real + 1j*t_emb_transf1_imag
#         t_emb_transf2_real = nn.Dense(
#             self.n_modes//2+1,
#             use_bias=False,
#         )(t_emb)
#         t_emb_transf2_imag = nn.Dense(
#             self.n_modes//2+1,
#             use_bias=False,
#         )(t_emb)
#         t_emb_transf2 = t_emb_transf2_real + 1j*t_emb_transf2_imag

#         weights1 = jnp.einsum("i,ijkl->ijkl", t_emb_transf1[:self.n_modes//2+1], weights1)
#         weights2 = jnp.einsum("i,ijkl->ijkl", t_emb_transf2[:self.n_modes//2+1], weights2)

#         x_ft1 = jnp.einsum('ijk,ijkl->ijl', x_ft[:self.n_modes//2+1, :self.n_modes//2+1, :], weights1)
#         x_ft2 = jnp.einsum('ijk,ijkl->ijl', x_ft[-(self.n_modes//2+1):, :self.n_modes//2+1, :], weights2)

#         out_ft = out_ft.at[:self.n_modes//2+1, :self.n_modes//2+1, :].set(x_ft1)
#         out_ft = out_ft.at[-(self.n_modes//2+1):, :self.n_modes//2+1, :].set(x_ft2)

#         out = jnp.fft.irfft2(out_ft, s=(out_grid_sz, out_grid_sz), axes=(0, 1), norm=self.fft_norm)
#         return out
class SpectralFreqTimeConv2D(nn.Module):
    """ Time modulated integral kernel operator with symmetry fix 
    Modified to handle both upper and lower frequency blocks similar to PyTorch implementation
    """
    in_co_dim: int
    out_co_dim: int
    t_emb_dim: int
    n_modes: tuple  # Changed to tuple for separate x,y modes
    out_grid_sz: int = None
    fft_norm: str = "forward"
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray) -> jnp.ndarray:
        """ 
        x shape: (in_grid_sz, in_grid_sz, in_co_dim),
        t_emb shape: (t_emb_dim)
        output shape: (out_grid_sz, out_grid_sz, out_co_dim)
        """
        in_grid_sz_h, in_grid_sz_w = x.shape[0], x.shape[1]
        out_grid_sz_h = in_grid_sz_h if self.out_grid_sz is None else self.out_grid_sz
        out_grid_sz_w = in_grid_sz_w if self.out_grid_sz is None else self.out_grid_sz
        
        # Get half modes for each dimension
        half_n_modes = (self.n_modes[0] // 2, self.n_modes[1] // 2)
        
        # 参数初始化（上下块各一个权重矩阵）-------------------------------------------------
        # Upper block weights
        weights_upper_real = self.param(
            'weights_upper_real',
            nn.initializers.normal(stddev=1/(self.in_co_dim + self.out_co_dim)),
            (half_n_modes[0], half_n_modes[1], self.in_co_dim, self.out_co_dim)
        )
        weights_upper_imag = self.param(
            'weights_upper_imag',
            nn.initializers.normal(stddev=1/(self.in_co_dim + self.out_co_dim)),
            (half_n_modes[0], half_n_modes[1], self.in_co_dim, self.out_co_dim)
        )
        weights_upper = weights_upper_real + 1j * weights_upper_imag
        
        # Lower block weights
        weights_lower_real = self.param(
            'weights_lower_real',
            nn.initializers.normal(stddev=1/(self.in_co_dim + self.out_co_dim)),
            (half_n_modes[0], half_n_modes[1], self.in_co_dim, self.out_co_dim)
        )
        weights_lower_imag = self.param(
            'weights_lower_imag',
            nn.initializers.normal(stddev=1/(self.in_co_dim + self.out_co_dim)),
            (half_n_modes[0], half_n_modes[1], self.in_co_dim, self.out_co_dim)
        )
        weights_lower = weights_lower_real + 1j * weights_lower_imag
        
        # 时间嵌入调制 ------------------------------------------------------------
        # 为上下块分别生成调制因子
        t_projection = nn.Dense(
            features=4*half_n_modes[0]*half_n_modes[1],  # 上下块的实部和虚部
            use_bias=False
        )(t_emb)
        
        # 分割为上块和下块的调制因子
        t_upper = t_projection[:2*half_n_modes[0]*half_n_modes[1]]
        t_lower = t_projection[2*half_n_modes[0]*half_n_modes[1]:]
        
        # 上块的实部和虚部
        t_upper_real = t_upper[::2]
        t_upper_imag = t_upper[1::2]
        
        # 下块的实部和虚部
        t_lower_real = t_lower[::2]
        t_lower_imag = t_lower[1::2]
        
        # 重塑调制因子形状与权重矩阵匹配
        modulation_upper = jnp.reshape(
            t_upper_real + 1j*t_upper_imag, 
            (half_n_modes[0], half_n_modes[1], 1, 1)
        )
        
        modulation_lower = jnp.reshape(
            t_lower_real + 1j*t_lower_imag,
            (half_n_modes[0], half_n_modes[1], 1, 1)
        )
        
        # 应用时间调制
        modulated_weights_upper = weights_upper * modulation_upper
        modulated_weights_lower = weights_lower * modulation_lower
        
        # 傅里叶变换处理 ---------------------------------------------------------
        x_ft = jnp.fft.rfft2(x, axes=(0, 1), norm=self.fft_norm)
        
        # 创建输出频谱
        out_ft = jnp.zeros(
            (in_grid_sz_h, in_grid_sz_w//2 + 1, self.out_co_dim),
            dtype=jnp.complex64
        )
        
        # 处理上块（低频部分）
        x_ft_upper = x_ft[:half_n_modes[0], :half_n_modes[1], :]
        out_ft_upper = jnp.einsum('ijk,ijkl->ijl', x_ft_upper, modulated_weights_upper)
        
        # 处理下块（高频部分）
        x_ft_lower = x_ft[-half_n_modes[0]:, :half_n_modes[1], :]
        out_ft_lower = jnp.einsum('ijk,ijkl->ijl', x_ft_lower, modulated_weights_lower)
        
        # 将结果填入输出频谱
        out_ft = out_ft.at[:half_n_modes[0], :half_n_modes[1], :].set(out_ft_upper)
        out_ft = out_ft.at[-half_n_modes[0]:, :half_n_modes[1], :].set(out_ft_lower)
        
        # 逆变换恢复空间信号 ------------------------------------------------------
        out = jnp.fft.irfft2(
            out_ft, 
            s=(out_grid_sz_h, out_grid_sz_w),
            axes=(0, 1), 
            norm=self.fft_norm
        )
        
        # 可选: 添加偏置项
        # bias = self.param(
        #     'bias',
        #     nn.initializers.zeros,
        #     (self.out_co_dim,)
        # )
        # # out = out + bias
        
        return out
# class SpectralFreqTimeConv2D(nn.Module):
#     """ Time modulated integral kernel operator with symmetry fix """
#     in_co_dim: int
#     out_co_dim: int
#     t_emb_dim: int
#     n_modes: int
#     out_grid_sz: int = None
#     fft_norm: str = "forward"

#     @nn.compact
#     def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray) -> jnp.ndarray:
#         """ x shape: (in_grid_sz, in_grid_sz, in_co_dim),
#             t_emb shape: (t_emb_dim)
#             output shape: (out_grid_sz, out_grid_sz, out_co_dim)
#         """
#         in_grid_sz = x.shape[0]
#         out_grid_sz = in_grid_sz if self.out_grid_sz is None else self.out_grid_sz

#         # 参数初始化（单一权重矩阵） -------------------------------------------------
#         weights_shape = (self.n_modes//2 + 1, self.n_modes//2 + 1, 
#                         self.in_co_dim, self.out_co_dim)
        
#         # 基础权重参数
#         weights_real = self.param(
#             'weights_real',
#             nn.initializers.normal(stddev=1/(self.in_co_dim + self.out_co_dim)),
#             weights_shape
#         )
#         weights_imag = self.param(
#             'weights_imag',
#             nn.initializers.normal(stddev=1/(self.in_co_dim + self.out_co_dim)),
#             weights_shape
#         )
#         base_weights = weights_real + 1j * weights_imag

#         # 时间嵌入调制 ------------------------------------------------------------
#         # 生成复数调制因子 (形状: (n_modes//2+1, 1, 1, 1))
#         t_projection = nn.Dense(
#             features=2*(self.n_modes//2 + 1),  # 同时生成实部和虚部
#             use_bias=False
#         )(t_emb)
#         t_real = t_projection[::2]  # 取偶数索引作为实部
#         t_imag = t_projection[1::2]  # 取奇数索引作为虚部
#         modulation = (t_real + 1j*t_imag)[:self.n_modes//2 + 1]  # 截取有效部分
#         modulation = jnp.expand_dims(modulation, axis=(1,2,3))  # 添加额外维度

#         # 应用时间调制
#         modulated_weights = base_weights * modulation

#         # 傅里叶变换处理 ---------------------------------------------------------
#         x_ft = jnp.fft.rfft2(x, axes=(0, 1), norm=self.fft_norm)

#         # 仅处理正频率的低频区域
#         x_ft_trunc = x_ft[:self.n_modes//2 + 1, :self.n_modes//2 + 1, :]
#         out_ft_upper = jnp.einsum('ijk,ijkl->ijl', x_ft_trunc, modulated_weights)


#         # 构建输出频谱
#         out_ft = jnp.zeros((in_grid_sz, in_grid_sz//2 + 1, self.out_co_dim),
#                           dtype=jnp.complex64)
#         out_ft = out_ft.at[:self.n_modes//2 + 1, :self.n_modes//2 + 1, :].set(out_ft_upper)

#         # 逆变换恢复空间信号 ------------------------------------------------------
#         out = jnp.fft.irfft2(out_ft, s=(out_grid_sz, out_grid_sz), 
#                             axes=(0, 1), norm=self.fft_norm)
#         return out.real  # 确保输出为实数
    
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
            x = jax.image.resize(x, (self.out_grid_sz, self.out_co_dim), method="bicubic")
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
            x = jax.image.resize(x, (self.out_grid_sz, self.out_grid_sz, self.out_co_dim), method="bicubic")
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
        if self.norm.lower() == "instance":
            x_out = nn.LayerNorm()(x_out)

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
        if self.norm.lower() == "instance":
            x_out = nn.LayerNorm()(x_out)

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


class ChannelMLP(nn.Module):
    """ChannelMLP applies an arbitrary number of layers of 
    1d convolution and nonlinearity to the channels of input
    and is invariant to spatial resolution.
    
    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is nn.gelu
    dropout : float, default is 0
        if > 0, dropout probability
    """
    in_channels: int
    out_channels: Optional[int] = None
    hidden_channels: Optional[int] = None
    n_layers: int = 2
    n_dim: int = 2
    activation: str = "gelu"
    dropout: float = 0.0
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Initialize parameters
        out_channels = self.in_channels if self.out_channels is None else self.out_channels
        hidden_channels = self.in_channels if self.hidden_channels is None else self.hidden_channels
        
        reshaped = False
        size = list(x.shape)
        if len(size) > 3:  
            # batch, channels, x1, x2... extra dims
            x = jnp.reshape(x, (size[0], size[1], -1))
            reshaped = True
        
        # In Flax, we define layers in the __call__ method with self.variable decorators
        for i in range(self.n_layers):
            if i == 0 and i == (self.n_layers - 1):
                x = nn.Conv(out_channels, kernel_size=(1,), name=f'conv_{i}')(x)
            elif i == 0:
                x = nn.Conv(hidden_channels, kernel_size=(1,), name=f'conv_{i}')(x)
            elif i == (self.n_layers - 1):
                x = nn.Conv(out_channels, kernel_size=(1,), name=f'conv_{i}')(x)
            else:
                x = nn.Conv(hidden_channels, kernel_size=(1,), name=f'conv_{i}')(x)
            
            if i < self.n_layers - 1:
                x = get_activation_fn(self.activation)(x)
            
            if self.dropout > 0.0:
                x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        
        # If x was an N-d tensor reshaped into 1d, undo the reshaping
        if reshaped:
            new_shape = (size[0], out_channels) + tuple(size[2:])
            x = jnp.reshape(x, new_shape)
            
        return x


class LinearChannelMLP(nn.Module):
    """Simple MLP with Dense Layers."""
    layers: tuple
    activation: str = "relu"
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        activation_fn = get_activation_fn(self.activation)
        
        # Make sure x is at least 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # Get input dimension from x
        input_dim = x.shape[-1]
        
        # Create complete layer configuration
        if not self.layers:
            layers = [input_dim, 64, 32, 16]
        else:
            layers = list(self.layers)
            # Make sure first layer matches input dimension
            if layers[0] != input_dim:
                layers = [input_dim] + layers[1:]
        
        # Apply dense layers with fixed parameter path
        for i, features in enumerate(layers[1:]):
            x = nn.Dense(
                features=features,
                use_bias=self.use_bias,
                kernel_init=nn.initializers.normal(0.02),
                name=f'dense_{i}'
            )(x)
            
            # Apply activation except for the last layer
            if i < len(layers) - 2:
                x = activation_fn(x)
                
        return x

# integral kernel transform for GNO

class IntegralTransform(nn.Module):
    kernel_mlp_layers: tuple
    kernel_mlp_activation: str = "gelu"
    transform_type: str = "linear" 

    '''
    Computes one of the following:
        (a) \int_{A(x)} k(x, y) dy
        (b) \int_{A(x)} k(x, y) * f(y) dy  # default
        (c) \int_{A(x)} k(x, y, f(y)) dy
        (d) \int_{A(x)} k(x, y, f(y)) * f(y) dy

    kernel is parametrized by a MLP

    kernel_mlp is a predefined MLP, if None, a new MLP will be created 
    by the LinearChannelMLP class given the kernel_mlp_layers and kernel_mlp_activation
    kernel_mlp_layers is a list of layers sizes for the MLP
    kernel_mlp_activation is the activation function for the MLP

    transform_type : str, default 'linear'
    Which integral transform to compute. The mapping is:
    'linear_kernelonly' -> (a)
    'linear' -> (b)
    'nonlinear_kernelonly' -> (c)
    'nonlinear' -> (d)
    If the input f is not given then (a) is computed
    '''

    def _process_neighbors(self, y: jnp.ndarray, neibor_mask: jnp.ndarray, f_y: jnp.ndarray = None) -> jnp.ndarray:
        # neibor_mask shape (n_x, n_y), csr matrix
        # y shape (n_y, dim_y)
        # f_y shape (n_y, dim_f)
        num_neighbors = neibor_mask.indptr[1:] - neibor_mask.indptr[:-1] # (n_x,)
        # neibored_sum shape (sum_n_neighbors,)
        print(neibor_mask.shape)
        print(y.shape)

        y_neighbored = y[neibor_mask.indices]
        if f_y is not None:
            f_y_neighbored = f_y[neibor_mask.indices]
        else:
            f_y_neighbored = None
            
        return y_neighbored, f_y_neighbored, num_neighbors
    
    @nn.compact
    def __call__(self, y: jnp.ndarray, neibor_mask: jnp.ndarray, x: jnp.ndarray = None, f_y: jnp.ndarray = None, weights: jnp.ndarray = None) -> jnp.ndarray:
        '''
        y: the input points, shape (n_y, dim_y), the space of integration
        x: the evalueate points for the new function, shape (n_x, dim_x)
        neibor_mask: the neighbors of x, shape (n_x, n_y)
        f_y: the values of the input function at y, shape (n_y, dim_f) dim_f should 
        be the same as the output of the kernel MLP(out_channels), if not, force to compute (a)

        weights: the weights of the input function at y, shape (n_y, n_neighbors)
        
        '''
        
        kernel_mlp = LinearChannelMLP(self.kernel_mlp_layers, self.kernel_mlp_activation)
        if x is None:
            x = y
        # check the shape of f_y
        if f_y is not None and f_y.shape[1] != kernel_mlp.layers[-1]:
            raise ValueError(f"f_y shape (n_nodes, dim_f) should be the same as the output of the kernel MLP(out_channels), got {f_y.shape[1]} and {kernel_mlp.layers[-1]}")
        # find the num of neighbors for each output grid point
        # n_neighbors = jnp.array([len(neighbors_indices[i]) for i in range(len(neighbors_indices))]) # (n_points,)
        # repeat the x for each neighbor
        y_neighbored, f_y_neighbored, n_neighbors = self._process_neighbors(y, neibor_mask, f_y)
        x = jnp.repeat(x, n_neighbors, axis=0) # (total_neighbors, dim_x)
        # only integrate over the neighbors of x so we only need corresponding neighbor y and f_y
        # get the neighbored y of each x(output grid points) 
        total_neighbors = sum(n_neighbors)
        
        # concatenate x, neighbors, f_y conditionally
        if f_y is not None and (self.transform_type == 'nonlinear' or self.transform_type == 'nonlinear_kernelonly'):
            kernel_input = jnp.concatenate([y_neighbored, f_y], axis=-1) # (y, f_y), (total_neighbors, dim_y + dim_f)
            kernel_input = jnp.concatenate([kernel_input, x], axis=-1)  # (x, y, f_y), (total_neighbors, dim_x + dim_y + dim_f) 
        else:
            kernel_input = jnp.concatenate([x, y_neighbored], axis=-1)  # (x, y), (total_neighbors, dim_x + dim_y)

        # apply the kernel MLP
        kernel_output = kernel_mlp(kernel_input) # (total_neighbors, out_channels)
        

        # compute the integral transform
        if self.transform_type == 'linear' or self.transform_type == 'nonlinear':
            # times the neighbored f_y(total_neighbors, dim_f) and kernel_output (total_neighbors, out_channels)
            
            if f_y.shape[1] == kernel_output.shape[1]:
                result = kernel_output * f_y_neighbored
            elif f_y.shape[1] == 1:
                result = kernel_output * f_y_neighbored
            # else:
            #     result = jnp.einsum("ij,ik->ijk", kernel_output, f_y_neighbored) # (total_neighbors, out_channels, dim_f)
            #     # sum over the neighbors
            #     result = jnp.sum(result, axis=-1) # (total_neighbors, out_channels)
            # times dy
            result = segment_csr(result, neibor_mask.indptr)

        elif self.transform_type == 'linear_kernelonly' or self.transform_type == 'nonlinear_kernelonly':
            # kernel_output shape (total_neighbors, out_channels)
            # sum over the neighbors
            # times dy
            result = segment_csr(kernel_output, neibor_mask.indptr)

        else:
            raise ValueError(f"Invalid transform type: {self.transform_type}")

        return result
        
        
        
        
        
        
        



# Graph neural Operator blocks

class GNOBlock(nn.Module):
    in_co_dim: int = 3 # number of channels for input points y and x default is 3
    in_channels: int = None # number of channels in input function f_y. Only used if transform_type
                        # is (c) "nonlinear" or (d) "nonlinear_kernelonly", None by default
    out_channels: int = in_co_dim # number of channels in output function default is 1
    radius: float = 0.1 # radius of the neighbors
    kernel_mlp_layers: tuple = (128, 128, 128)
    kernel_mlp_activation: str = "gelu" # activation function for the kernel MLP
    transform_type: str = "linear_kernelonly" # type of integral transform to compute

    @nn.compact
    def __call__(self, y: jnp.ndarray, x: jnp.ndarray = None, f_y: jnp.ndarray = None) -> jnp.ndarray:
        '''
        y shape: (n_y, in_co_dim) previous layer output, integration domain
        x shape: (n_x, in_co_dim) current layer grid points, evaluation domain
        x and y should have the same in_co_dim(in same domain)
        f_y shape: (n_y, in_channels) previous layer grid points values, 
        function to integrate the kernel against defined on the points y. 
        The kernel is assumed diagonal, so the in_channels should be the same as the output of the kernel MLP(out_channels)
        '''
        neibor_mask = find_nearest_neighbors_in_ball(x, y, self.radius, sparse=True)
        # compute the integral transform
        if self.in_channels is None:
            transform_type = "linear_kernelonly"
        integral_transform = IntegralTransform(kernel_mlp_layers=self.kernel_mlp_layers, kernel_mlp_activation=self.kernel_mlp_activation, transform_type=self.transform_type)
        integral_output = integral_transform(y, neibor_mask, x, f_y)
        

        return integral_output


# class IntegralTransform(nn.Module):
#     kernel_mlp: Optional[nn.Module] = None
#     kernel_mlp_layers: list = field(default_factory=list)
#     kernel_mlp_activation: str = "gelu"
#     transform_type: str = "linear" 

#     '''
#     Computes one of the following:
#         (a) \int_{A(x)} k(x, y) dy
#         (b) \int_{A(x)} k(x, y) * f(y) dy  # default
#         (c) \int_{A(x)} k(x, y, f(y)) dy
#         (d) \int_{A(x)} k(x, y, f(y)) * f(y) dy
#     '''
    
#     @nn.compact
#     def __call__(self, y: jnp.ndarray, neighbors_mask: jnp.ndarray, x: Optional[jnp.ndarray] = None, 
#                  f_y: Optional[jnp.ndarray] = None, weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
#         '''
#         极简化版，完全重写为矩阵运算，避免使用动态控制流
        
#         Args:
#             y: 输入点，形状 (n_y, dim_y)
#             neighbors_mask: 邻居掩码，形状 (n_x, n_y)，其中True表示y[j]是x[i]的邻居
#             x: 评估点，形状 (n_x, dim_x)，如果为None，则等于y
#             f_y: 输入函数在y点的值，形状 (n_y, dim_f)
        
#         Returns:
#             结果矩阵，形状 (n_x, out_channels)
#         '''
#         # 如果x为None，设置为y
#         if x is None:
#             x = y
        
#         # 计算输入维度，以确保MLP层数组正确
#         dim_x = x.shape[1]
#         dim_y = y.shape[1]
        
#         # 根据transform_type计算输入维度
#         if f_y is not None and (self.transform_type == 'nonlinear' or self.transform_type == 'nonlinear_kernelonly'):
#             dim_f = f_y.shape[1]
#             input_dim = dim_x + dim_y + dim_f
#         else:
#             input_dim = dim_x + dim_y
        
#         # 确保kernel_mlp_layers包含输入维度
#         mlp_layers = list(self.kernel_mlp_layers)
#         if not mlp_layers:
#             # If empty, create a default structure with input_dim and output_dim=1
#             mlp_layers = [input_dim, input_dim//2, 1]
#         elif mlp_layers[0] != input_dim:
#             # If first layer doesn't match input_dim, prepend it
#             mlp_layers = [input_dim] + mlp_layers
        
#         # 初始化MLP - must provide complete layer dimensions including input and output
#         kernel_mlp = self.kernel_mlp if self.kernel_mlp is not None else LinearChannelMLP(
#             layers=mlp_layers,
#             activation=self.kernel_mlp_activation
#         )
        
#         n_x, n_y = neighbors_mask.shape
#         out_channels = mlp_layers[-1]
        
#         # 计算每个点的邻居数量
#         n_neighbors = jnp.sum(neighbors_mask, axis=1)  # (n_x,)
        
#         # 重塑x和y为合适的形状进行广播
#         x_expanded = x[:, None, :]  # (n_x, 1, dim_x)
#         y_expanded = y[None, :, :]  # (1, n_y, dim_y)
        
#         # 对于每个(x_i, y_j)对，准备输入到核MLP
#         # 复制x到(n_x, n_y, dim_x)
#         x_repeated = jnp.broadcast_to(x_expanded, (n_x, n_y, x.shape[1]))
#         # 复制y到(n_x, n_y, dim_y)
#         y_repeated = jnp.broadcast_to(y_expanded, (n_x, n_y, y.shape[1]))
        
#         # 如果f_y不为None，也将其广播
#         if f_y is not None:
#             f_y_expanded = f_y[None, :, :]  # (1, n_y, dim_f)
#             f_y_repeated = jnp.broadcast_to(f_y_expanded, (n_x, n_y, f_y.shape[1]))
        
#         # 构建核函数输入
#         if f_y is not None and (self.transform_type == 'nonlinear' or self.transform_type == 'nonlinear_kernelonly'):
#             # 包含函数值
#             kernel_inputs = jnp.concatenate([
#                 x_repeated.reshape(n_x * n_y, -1),
#                 y_repeated.reshape(n_x * n_y, -1),
#                 f_y_repeated.reshape(n_x * n_y, -1)
#             ], axis=1)
#         else:
#             # 仅包含坐标
#             kernel_inputs = jnp.concatenate([
#                 x_repeated.reshape(n_x * n_y, -1),
#                 y_repeated.reshape(n_x * n_y, -1)
#             ], axis=1)
            
#         # Explicitly check dimensions before calling kernel_mlp
#         expected_input_dim = kernel_inputs.shape[-1]
#         if mlp_layers[0] != expected_input_dim:
#             raise ValueError(f"MLP input dimension mismatch: expected {expected_input_dim}, got {mlp_layers[0]}")
        
#         # 应用核MLP
#         kernel_outputs = kernel_mlp(kernel_inputs)  # (n_x * n_y, out_channels)
        
#         # 将输出重塑回(n_x, n_y, out_channels)
#         kernel_outputs = kernel_outputs.reshape(n_x, n_y, out_channels)
        
#         # 应用掩码 - 将非邻居对应的输出设为0
#         mask_expanded = neighbors_mask[:, :, None]  # (n_x, n_y, 1)
#         masked_outputs = kernel_outputs * mask_expanded  # (n_x, n_y, out_channels)
        
#         # 计算积分变换
#         if self.transform_type == 'linear' or self.transform_type == 'nonlinear':
#             if f_y is not None:
#                 # 对于linear变换，还需要乘以f_y
#                 masked_outputs = masked_outputs * f_y_repeated
            
#             # 对邻居求和
#             result = jnp.sum(masked_outputs, axis=1)  # (n_x, out_channels)
            
#             # 归一化
#             n_neighbors_safe = jnp.maximum(n_neighbors, jnp.ones_like(n_neighbors))
#             result = result / n_neighbors_safe[:, None]
            
#         elif self.transform_type == 'linear_kernelonly' or self.transform_type == 'nonlinear_kernelonly':
#             # 仅对邻居求和
#             result = jnp.sum(masked_outputs, axis=1)  # (n_x, out_channels)
            
#             # 归一化
#             n_x_safe = jnp.maximum(n_x, 1)
#             result = result / n_x_safe
            
#         else:
#             # 默认行为
#             result = jnp.sum(masked_outputs, axis=1)  # (n_x, out_channels)
#             result = result / jnp.maximum(jnp.sum(neighbors_mask), 1)
        
#         return result

# class GNOBlock(nn.Module):
#     in_co_dim: int = 3  # 输入点y和x的通道数，默认为3
#     in_channels: Optional[int] = None  # 输入函数f_y的通道数
#     out_channels: int = None  # 输出函数的通道数
#     radius: float = 0.1  # 邻居搜索半径
#     kernel_mlp_layers: list = field(default_factory=list)  # 核MLP的层大小
#     kernel_mlp_activation: str = "gelu"  # 核MLP的激活函数
#     transform_type: str = "linear_kernelonly"  # 积分变换类型

#     @nn.compact
#     def __call__(self, y: jnp.ndarray, x: Optional[jnp.ndarray] = None, f_y: Optional[jnp.ndarray] = None) -> jnp.ndarray:
#         '''
#         y: 形状 (n_y, in_co_dim) 上一层输出，积分域
#         x: 形状 (n_x, in_co_dim) 当前层网格点，评估域
#         f_y: 形状 (n_y, in_channels) 上一层网格点的函数值
#         '''
#         if self.out_channels is None:
#             out_channels = self.in_co_dim
#         else:
#             out_channels = self.out_channels
            
#         if x is None:
#             x = y
            
#         # 使用find_nearest_neighbors_in_ball查找邻居
#         # 假设已经导入了函数
#         neighbors_mask_csr = find_nearest_neighbors_in_ball(x, y, self.radius, sparse=True)
        
#         # 计算积分变换
#         if self.in_channels is None:
#             transform_type = "linear_kernelonly"
#         else:
#             transform_type = self.transform_type
            
#         integral_transform = IntegralTransform(
#             kernel_mlp_layers=self.kernel_mlp_layers, 
#             kernel_mlp_activation=self.kernel_mlp_activation, 
#             transform_type=transform_type
#         )
        
#         integral_output = integral_transform(y, neighbors_mask_csr, x, f_y)
        
#         return integral_output

    
class GICTUNOBlock(nn.Module):
    in_co_dim: int = 3  # number of channels for input points y and x default is 3
    in_channels: int = None  # number of channels in input function f_y
    out_channels: int = None  # number of channels in output function 
    radius: float = 0.1  # radius of the neighbors
    kernel_mlp_layers: tuple = (128, 128, 128)  # layers of the kernel MLP
    kernel_mlp_activation: str = "gelu"  # activation function for the kernel MLP
    transform_type: str = "linear_kernelonly"  # type of integral transform to compute
    
    @nn.compact
    def __call__(self, y: jnp.ndarray, time_emb: jnp.ndarray, x: jnp.ndarray = None, f_y: jnp.ndarray = None) -> jnp.ndarray:
        '''
        Call GNOBlock and then apply a normalization both on time and space
        x is the latent geometry grid points, y is the integration domain 
        y is the input geometry, shape (n_y, dim_y), the space of integration
        output shape: (n_x, out_channels)
        '''
        # Determine output channels
        out_channels = self.out_channels if self.out_channels is not None else self.in_co_dim
        
        # Ensure kernel_mlp_layers has the correct output dimension
        # mlp_layers = list(self.kernel_mlp_layers)
        # if not mlp_layers:
        #     # Default layers if none provided
        #     mlp_layers = [y.shape[1] + (x.shape[1] if x is not None else y.shape[1]), 128, 64, out_channels]
        # elif mlp_layers[-1] != out_channels:
        #     # Adjust last layer to match out_channels
        #     mlp_layers[-1] = out_channels
        
        # Apply the GNO block with corrected mlp_layers
        if self.kernel_mlp_layers[-1] != out_channels:
            mlp_layers = self.kernel_mlp_layers + (out_channels,)
        else:
            mlp_layers = self.kernel_mlp_layers
        geo_output = GNOBlock(
            in_co_dim=self.in_co_dim, 
            in_channels=self.in_channels, 
            out_channels=out_channels,  # Ensure this matches TimeConv1D
            radius=self.radius, 
            kernel_mlp_layers=mlp_layers,  # Use adjusted layers
            kernel_mlp_activation=self.kernel_mlp_activation, 
            transform_type=self.transform_type
        )(y, x, f_y)
        
        # Apply a time-dependent convolution with matching out_channels
        time_residual = TimeConv1D(
            out_co_dim=out_channels,  # Match GNOBlock output channels
            out_grid_sz=geo_output.shape[0]
        )(y, time_emb)
        
        # Double check shapes match before addition
        if geo_output.shape != time_residual.shape:
            # Reshape if needed - this is a fallback
            print(f"WARNING: Reshaping outputs to match: geo_output {geo_output.shape} vs time_residual {time_residual.shape}")
            if geo_output.shape[0] == time_residual.shape[0]:
                # Only channel dimension mismatch, adapt with a Dense layer
                time_residual = nn.Dense(geo_output.shape[1])(time_residual)
            
        # Now combine the outputs
        output = time_residual + geo_output
        
        # Apply normalization
        output = nn.LayerNorm()(output)
        return output
