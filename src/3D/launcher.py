import polyscope as ps
import numpy as np
import jax.numpy as jnp
import igl
import os

ps.init()
cwd = os.getcwd()
v, f = igl.read_triangle_mesh(cwd + '/data/meshes/bunny.obj')
vertices = v
faces = f

# visualize!
ps_mesh = ps.register_surface_mesh("bunny", vertices, faces)
ps.show()