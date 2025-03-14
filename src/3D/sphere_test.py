
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import polyscope.imgui as psim
import igl
import scipy as sp
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.nn as nn
import jax.lax as lax
import jax.random as jrandom
import jax.nn as nn
import jax.lax as lax
import os
import jax.experimental.sparse as jsp
from scipy.sparse import diags, csr_matrix
from src.data.ToyData import *
from src.SDE import *
from src.SDESolver import *
import random
import trimesh
import polyscope as ps


def get_random_int():
    return random.randint(0, 1000000)
cwd = os.getcwd()
# v, f = igl.read_triangle_mesh(cwd + '/data/meshes/bunny.obj')
# K = igl.cotmatrix(v, f)

# M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)

# m_inv = 1.0 / M.diagonal()
# M_inv = diags(m_inv, format="csr")

# L = -M_inv @ K
def sphere_test():
    sphere_data_generator = SphereDataGenerator(landmark_num=500, radius=1.0, center=jnp.array([0.0, 0.0, 0.0]), seed=0)

    x0 = sphere_data_generator.generate_data(500, 1)
    x0 = x0[0]
    sde_3d = Kunita_Flow_SDE_3D_Eulerian(k_alpha=2.0, k_sigma=0.5, grid_num=25, grid_range=[-1,1], x0=x0)
    sde_solver = EulerMaruyama.from_sde(sde_3d, 0.01, 1.0, 3, None,debug_mode=False)
    xs,_ = sde_solver.solve(x0, rng_key=jrandom.PRNGKey(get_random_int()))
    return xs, x0

if __name__ == "__main__":
    xs, x0 = sphere_test()
    x0 = np.array(x0)
    xs = np.array(xs)
    print(x0.shape)
    print(xs.shape)
    ps.init()
    # global frame_idx
    frame_idx = 10
    def active_animation():
        for x in xs:
            ps_cloud = ps.register_point_cloud("my points", x)

            # ps_mesh.add_scalar_quantity("scalar", xs[:, 0], enabled=True)



    def imgui_callback():
        global frame_idx
        
        ps_cloud = ps.register_point_cloud("my points", x0)


        changed, frame_idx = psim.SliderFloat("Frame_idx", frame_idx, v_min=0,v_max=len(xs)-1)

        if changed:
            ps.remove_all_structures()
            frame_idx = int(frame_idx)
            ps_cloud = ps.register_point_cloud("my points", xs[frame_idx])
        


    ps.set_user_callback(imgui_callback)
    ps.show()

