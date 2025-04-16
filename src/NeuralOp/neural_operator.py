from dataclasses import field
from flax import linen as nn

from .blocks import *
import math
class CTUNO1D(nn.Module):
    """ U-Net shaped time-dependent neural operator"""
    out_co_dim: int
    lifting_dim: int
    co_dims_fmults: tuple
    n_modes_per_layer: tuple
    norm: str = "instance"
    act: str  = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (in_grid_sz, in_co_dim)
            t shape: (,)
            output shape: (out_grid_sz, out_co_dim)
        """
        t_emb_dim = 4 * self.lifting_dim
        in_grid_sz = x.shape[0]
        co_dims_fmults = (1,) + self.co_dims_fmults

        t_emb = TimeEmbedding(
            t_emb_dim,
        )(t)

        x = nn.Dense(
            self.lifting_dim,
        )(x)

        out_grid_sz_fmults = [1. / dim_fmult for dim_fmult in co_dims_fmults]

        downs = []
        for idx_layer in range(len(self.co_dims_fmults)):
            in_co_dim_fmult = co_dims_fmults[idx_layer]
            out_co_dim_fmult = co_dims_fmults[idx_layer+1]
            out_grid_sz = int(out_grid_sz_fmults[idx_layer+1] * in_grid_sz)
            n_modes = self.n_modes_per_layer[idx_layer]
            x = CTUNOBlock1D(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act
            )(x, t_emb)
            downs.append(x)

        x = CTUNOBlock1D(
            in_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            out_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            t_emb_dim=t_emb_dim,
            n_modes=self.n_modes_per_layer[-1],
            out_grid_sz=int(out_grid_sz_fmults[-1] * in_grid_sz),
            norm=self.norm,
            act=self.act
        )(x, t_emb)

        for idx_layer in range(1, len(self.co_dims_fmults)+1):
            in_co_dim_fmult = co_dims_fmults[-idx_layer]
            out_co_dim_fmult = co_dims_fmults[-(idx_layer+1)] 
            out_grid_sz = int(out_grid_sz_fmults[-(idx_layer+1)] * in_grid_sz)
            n_modes = self.n_modes_per_layer[-idx_layer]
            down = downs[-idx_layer]
            x = jnp.concatenate([x, down], axis=-1)
            x = CTUNOBlock1D(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult * 2),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act
            )(x, t_emb)
        
        x = nn.Dense(
            self.out_co_dim,
        )(x)

        return x
    
class SphericalCTUNO1D(nn.Module):
    """ U-Net shaped time-dependent neural operator"""
    out_co_dim: int
    lifting_dim: int
    co_dims_fmults: tuple
    n_modes_per_layer: tuple
    norm: str = "instance"
    act: str  = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (in_grid_sz, in_co_dim)
            t shape: (,)
            output shape: (out_grid_sz, out_co_dim)
        """
        t_emb_dim = 4 * self.lifting_dim
        in_grid_sz = x.shape[0]
        co_dims_fmults = (1,) + self.co_dims_fmults

        t_emb = TimeEmbedding(
            t_emb_dim,
        )(t)

        x = nn.Dense(
            self.lifting_dim,
        )(x)

        out_grid_sz_fmults = [1. / dim_fmult for dim_fmult in co_dims_fmults]

        downs = []
        for idx_layer in range(len(self.co_dims_fmults)):
            in_co_dim_fmult = co_dims_fmults[idx_layer]
            out_co_dim_fmult = co_dims_fmults[idx_layer+1]
            out_grid_sz = int(out_grid_sz_fmults[idx_layer+1] * in_grid_sz)
            n_modes = self.n_modes_per_layer[idx_layer]
            x = SphericalCTUNOBlock1D(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act
            )(x, t_emb)
            downs.append(x)

        x = SphericalCTUNOBlock1D(
            in_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            out_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            t_emb_dim=t_emb_dim,
            n_modes=self.n_modes_per_layer[-1],
            out_grid_sz=int(out_grid_sz_fmults[-1] * in_grid_sz),
            norm=self.norm,
            act=self.act
        )(x, t_emb)

        for idx_layer in range(1, len(self.co_dims_fmults)+1):
            in_co_dim_fmult = co_dims_fmults[-idx_layer]
            out_co_dim_fmult = co_dims_fmults[-(idx_layer+1)] 
            out_grid_sz = int(out_grid_sz_fmults[-(idx_layer+1)] * in_grid_sz)
            n_modes = self.n_modes_per_layer[-idx_layer]
            down = downs[-idx_layer]
            x = jnp.concatenate([x, down], axis=-1)
            x = SphericalCTUNOBlock1D(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult * 2),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act
            )(x, t_emb)
        
        x = nn.Dense(
            self.out_co_dim,
        )(x)

        return x


class CTUNO2D(nn.Module):
    """ U-Net shaped time-dependent neural operator"""
    out_co_dim: int
    lifting_dim: int
    co_dims_fmults: tuple
    n_modes_per_layer: tuple
    norm: str = "instance"
    act: str  = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (in_grid_sz, in_grid_sz, in_co_dim)
            t shape: (,)
            output shape: (out_grid_sz, out_grid_sz, out_co_dim)
        """
                # check the x's shape, if x is 1D manifold, then we need to reshape x to 2D
        if x.ndim == 2:
            sqrt_n = int(math.sqrt(x.shape[0]))
            assert sqrt_n * sqrt_n == x.shape[0], "x.shape[0] is not a square number"
            x = jnp.reshape(x, (sqrt_n, sqrt_n, x.shape[1]))
            flatten = True
        else:
            flatten = False
        t_emb_dim = 4 * self.lifting_dim
        in_grid_sz = x.shape[0]
        co_dims_fmults = (1,) + self.co_dims_fmults

        t_emb = TimeEmbedding(
            t_emb_dim,
        )(t)

        x = nn.Conv(
            features=self.lifting_dim,
            kernel_size=(1, 1),
            padding="VALID"
        )(x)

        out_grid_sz_fmults = [1. / dim_fmult for dim_fmult in co_dims_fmults]

        downs = []
        for idx_layer in range(len(self.co_dims_fmults)):
            in_co_dim_fmult = co_dims_fmults[idx_layer]
            out_co_dim_fmult = co_dims_fmults[idx_layer+1]
            out_grid_sz = int(out_grid_sz_fmults[idx_layer+1] * in_grid_sz)
            n_modes = self.n_modes_per_layer[idx_layer]
            x = CTUNOBlock2D(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act
            )(x, t_emb)
            downs.append(x)

        x = CTUNOBlock2D(
            in_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            out_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            t_emb_dim=t_emb_dim,
            n_modes=self.n_modes_per_layer[-1],
            out_grid_sz=int(out_grid_sz_fmults[-1] * in_grid_sz),
            norm=self.norm,
            act=self.act
        )(x, t_emb)

        for idx_layer in range(1, len(self.co_dims_fmults)+1):
            in_co_dim_fmult = co_dims_fmults[-idx_layer]
            out_co_dim_fmult = co_dims_fmults[-(idx_layer+1)] 
            out_grid_sz = int(out_grid_sz_fmults[-(idx_layer+1)] * in_grid_sz)
            n_modes = self.n_modes_per_layer[-idx_layer]
            down = downs[-idx_layer]
            x = jnp.concatenate([x, down], axis=-1)
            x = CTUNOBlock2D(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult * 2),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act
            )(x, t_emb)
        
        x = nn.Conv(
            features=self.out_co_dim,
            kernel_size=(1, 1),
            padding="VALID"
        )(x)
        # flatten the x
        print("out x shape: ", x.shape)
        if flatten:
            x = jnp.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
        return x
        

class SphericalCTUNO2D(nn.Module):
    """ U-Net shaped time-dependent neural operator"""
    out_co_dim: int
    lifting_dim: int
    co_dims_fmults: tuple
    n_modes_per_layer: tuple
    norm: str = "instance"
    act: str  = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (in_grid_sz, in_grid_sz, in_co_dim)
            t shape: (,)
            output shape: (out_grid_sz, out_grid_sz, out_co_dim)
        """
                # check the x's shape, if x is 1D manifold, then we need to reshape x to 2D
        if x.ndim == 2:
            sqrt_n = int(math.sqrt(x.shape[0]))
            assert sqrt_n * sqrt_n == x.shape[0], "x.shape[0] is not a square number"
            x = jnp.reshape(x, (sqrt_n, sqrt_n, x.shape[1]))
            flatten = True
        else:
            flatten = False
        t_emb_dim = 4 * self.lifting_dim
        in_grid_sz = x.shape[0]
        co_dims_fmults = (1,) + self.co_dims_fmults

        t_emb = TimeEmbedding(
            t_emb_dim,
        )(t)

        x = nn.Conv(
            features=self.lifting_dim,
            kernel_size=(1, 1),
            padding="VALID"
        )(x)

        out_grid_sz_fmults = [1. / dim_fmult for dim_fmult in co_dims_fmults]

        downs = []
        for idx_layer in range(len(self.co_dims_fmults)):
            in_co_dim_fmult = co_dims_fmults[idx_layer]
            out_co_dim_fmult = co_dims_fmults[idx_layer+1]
            out_grid_sz = int(out_grid_sz_fmults[idx_layer+1] * in_grid_sz)
            n_modes = self.n_modes_per_layer[idx_layer]
            x = CTUNOBlock2D(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act
            )(x, t_emb)
            downs.append(x)

        x = CTUNOBlock2D(
            in_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            out_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            t_emb_dim=t_emb_dim,
            n_modes=self.n_modes_per_layer[-1],
            out_grid_sz=int(out_grid_sz_fmults[-1] * in_grid_sz),
            norm=self.norm,
            act=self.act
        )(x, t_emb)

        for idx_layer in range(1, len(self.co_dims_fmults)+1):
            in_co_dim_fmult = co_dims_fmults[-idx_layer]
            out_co_dim_fmult = co_dims_fmults[-(idx_layer+1)] 
            out_grid_sz = int(out_grid_sz_fmults[-(idx_layer+1)] * in_grid_sz)
            n_modes = self.n_modes_per_layer[-idx_layer]
            down = downs[-idx_layer]
            x = jnp.concatenate([x, down], axis=-1)
            x = CTUNOBlock2D(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult * 2),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act
            )(x, t_emb)
        
        x = nn.Conv(
            features=self.out_co_dim,
            kernel_size=(1, 1),
            padding="VALID"
        )(x)
        # flatten the x
        print("out x shape: ", x.shape)
        if flatten:
            x = jnp.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
        return x
        

class GICTUNO1D(nn.Module):
    """ Geometrically-informed conditioned U-Net shaped time-dependent neural operator"""
    in_co_dim: int
    out_co_dim: int
    latent_geo_grid_sz: tuple # (latent_geo_grid_sz_x, latent_geo_grid_sz_y, latent_geo_grid_sz_z)
    latent_geo_grid_range_x: tuple # (latent_geo_grid_range_x_min, latent_geo_grid_range_x_max)
    latent_geo_grid_range_y: tuple # (latent_geo_grid_range_y_min, latent_geo_grid_range_y_max)
    latent_geo_grid_range_z: tuple # (latent_geo_grid_range_z_min, latent_geo_grid_range_z_max)
    gno_radius: float
    gno_kernel_mlp_layers: tuple
    gno_transform_type: str
    lifting_dim: int
    co_dims_fmults: tuple
    n_modes_per_layer: tuple
    norm: str = "instance"
    act: str  = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """ x shape: (in_grid_sz, in_co_dim)
            t shape: (,)
            output shape: (out_grid_sz, out_co_dim)
        """
        t_emb_dim = 4 * self.lifting_dim
        in_grid_sz = x.shape[0]
        co_dims_fmults = (1,) + self.co_dims_fmults

        t_emb = TimeEmbedding(
            t_emb_dim,
        )(t)

        # generate the latent geometry grid points latent_geogrid (latent_geo_grid_sz, in_co_dim)
        # meshgrid the latent_geogrid
        latent_geogrid_x = jnp.linspace(self.latent_geo_grid_range_x[0], self.latent_geo_grid_range_x[1], self.latent_geo_grid_sz[0])
        latent_geogrid_y = jnp.linspace(self.latent_geo_grid_range_y[0], self.latent_geo_grid_range_y[1], self.latent_geo_grid_sz[1])
        latent_geogrid_z = jnp.linspace(self.latent_geo_grid_range_z[0], self.latent_geo_grid_range_z[1], self.latent_geo_grid_sz[2])
        latent_geogrid_x, latent_geogrid_y, latent_geogrid_z = jnp.meshgrid(latent_geogrid_x, latent_geogrid_y, latent_geogrid_z)
        latent_geogrid = jnp.stack([latent_geogrid_x, latent_geogrid_y, latent_geogrid_z], axis=-1)
        latent_geogrid = latent_geogrid.reshape(-1, 3)
        # apply the GNOBlock on the latent geometry grid points
        x = GICTUNOBlock(
            in_co_dim=self.in_co_dim,
            out_channels=self.lifting_dim,
            radius=self.gno_radius,
            kernel_mlp_layers=list(self.gno_kernel_mlp_layers),
            kernel_mlp_activation=self.act,
            transform_type=self.gno_transform_type
        )(x, t_emb, latent_geogrid)

        

        out_grid_sz_fmults = [1. / dim_fmult for dim_fmult in co_dims_fmults]

        downs = []
        for idx_layer in range(len(self.co_dims_fmults)):
            in_co_dim_fmult = co_dims_fmults[idx_layer]
            out_co_dim_fmult = co_dims_fmults[idx_layer+1]
            out_grid_sz = int(out_grid_sz_fmults[idx_layer+1] * in_grid_sz)
            n_modes = self.n_modes_per_layer[idx_layer]
            x = CTUNOBlock1D(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act
            )(x, t_emb)
            downs.append(x)

        x = CTUNOBlock1D(
            in_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            out_co_dim=self.lifting_dim * self.co_dims_fmults[-1],
            t_emb_dim=t_emb_dim,
            n_modes=self.n_modes_per_layer[-1],
            out_grid_sz=int(out_grid_sz_fmults[-1] * in_grid_sz),
            norm=self.norm,
            act=self.act
        )(x, t_emb)

        for idx_layer in range(1, len(self.co_dims_fmults)+1):
            in_co_dim_fmult = co_dims_fmults[-idx_layer]
            out_co_dim_fmult = co_dims_fmults[-(idx_layer+1)] 
            out_grid_sz = int(out_grid_sz_fmults[-(idx_layer+1)] * in_grid_sz)
            n_modes = self.n_modes_per_layer[-idx_layer]
            down = downs[-idx_layer]
            x = jnp.concatenate([x, down], axis=-1)
            x = CTUNOBlock1D(
                in_co_dim=int(self.lifting_dim * in_co_dim_fmult * 2),
                out_co_dim=int(self.lifting_dim * out_co_dim_fmult),
                t_emb_dim=t_emb_dim,
                n_modes=n_modes,
                out_grid_sz=out_grid_sz,
                norm=self.norm,
                act=self.act
            )(x, t_emb)
        x = nn.Dense(
            self.out_co_dim,
        )(x)
        # # meshgrid the output grid
        # out_grid_x = jnp.linspace(0, 1, self.out_grid_sz[0])
        # out_grid_y = jnp.linspace(0, 1, self.out_grid_sz[1])
        # out_grid_z = jnp.linspace(0, 1, self.out_grid_sz[2])
        # out_grid_x, out_grid_y, out_grid_z = jnp.meshgrid(out_grid_x, out_grid_y, out_grid_z, indexing='ij')
        # out_grid = jnp.stack([out_grid_x, out_grid_y, out_grid_z], axis=-1)
        # # apply the GNOBlock on the output as a decoder
        # x = GICTUNOBlock(
        #     in_co_dim=self.lifting_dim * self.co_dims_fmults[0],
        #     out_channels=self.out_co_dim,
        #     radius=self.gno_radius,
        #     kernel_mlp_layers=list(self.gno_kernel_mlp_layers),
        #     kernel_mlp_activation=self.act,
        #     transform_type=self.gno_transform_type
        # )(x, t_emb, out_grid)


        return x