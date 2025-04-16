
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import polyscope.imgui as psim
import scipy as sp
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.nn as nn
import jax.lax as lax
import os
from src.data.ToyData import *
from src.SDE import *
from src.SDESolver import *
import random
import polyscope as ps
from src.models import DsmModel
import src.Trainer as Trainer
import matplotlib.pyplot as plt
from src.plot import plot_trajectory_3d_polyscope, plot_trajectory_3d
from src.NeuralOp.neural_operator import CTUNO1D
from flax.training import checkpoints
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
    sde_3d = Kunita_Flow_SDE_3D_Eulerian(k_alpha=1.0, k_sigma=0.4, grid_num=25, grid_range=[-1,1], x0=x0)
    sde_solver = EulerMaruyama.from_sde(sde_3d, 0.01, 1.0, 3, None,debug_mode=False)
    xs,_ = sde_solver.solve(x0, rng_key=jrandom.PRNGKey(get_random_int()))
    return xs, x0

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))



if __name__ == "__main__":
    # xs, x0 = sphere_test()
    train_steps = 1000
    retrain = True
    retrain_steps = 1000

    grid_size = 12

    # sphere_data_generator_XT = SphereDataGenerator(landmark_num=500, radius=0.7, center=jnp.array([0.0, 0.0, 0.0]), seed=0)
    sphere_data_generator_XT = ManifoldDataGenerator2D(grid_size=grid_size, manifold_type="sphere", radius=0.7, flatten=False, seed = get_random_int())
    xT = sphere_data_generator_XT.generate_data(grid_size, 1)
    print(xT.shape)
    # sphere_data_generator_X0 = SphereDataGenerator(landmark_num=500, radius=0.5, center=jnp.array([0.0, 0.0, 0.0]), seed=0)
    sphere_data_generator_X0 = ManifoldDataGenerator2D(grid_size=grid_size, manifold_type="sphere", radius=0.5, flatten=False, seed = get_random_int())
    x0 = sphere_data_generator_X0.generate_data(grid_size, 5)
    print(x0.shape)
    # sde_3d = Kunita_Flow_SDE_3D_Eulerian_Optimized(k_alpha=1.6, k_sigma=0.4, grid_num=10, grid_range=[-1,1], x0=x0, batch_size=128)
    # sde_3d = Brownian_Motion_SDE_2Dmanifold(dim=3, sigma=0.1, x0=x0[0])
    sde_3d = Kunita_Flow_SDE_3D_Eulerian_2Dmanifold(k_alpha=1.6, k_sigma=0.4, grid_num=10, grid_range=[-1,1], x0=x0[0])
    sde_solver = EulerMaruyama.from_sde(sde_3d, 0.01, 1.0, 3, None,debug_mode=False)
    xs,_ = sde_solver.solve(x0[0], rng_key=jrandom.PRNGKey(get_random_int()))
    print(xs.shape)
    

    # dsm model
    model = DsmModel(dim=3, score_hidden_dims=(512, 512, 256), x_hidden_dims=(512, 512, 256), t_hidden_dims=(512, 512, 256), with_x0=True, t_embedding_dim=50)
    trainer = Trainer.SsmTrainer(seed=get_random_int(), landmark_num=grid_size)
    

    checkpoint_path = project_root() + '/checkpoints/sphere_model' + str(trainer.landmark_num)
    retrain_checkpoint_path = project_root() + '/checkpoints/sphere_model_retrain' + str(trainer.landmark_num)
    if not os.path.exists(checkpoint_path):
        train_state = trainer.train_state_init(model, lr=1e-4, model_kwargs={'x': jax.random.normal(jrandom.PRNGKey(get_random_int()), x0[0].shape), 't': jnp.array([0]), 'x0': x0[0], 'object_fn': 'Heng'})
        train_state, train_loss = trainer.train(train_state, sde_3d, sde_solver, sphere_data_generator_X0, train_steps, 6)
        plt.plot(train_loss)
        plt.show()
        # save the model
        config = {"dimension": x0[0].shape}
        ckpt = {"model": train_state, "config": config}
        checkpoints.save_checkpoint(checkpoint_path, ckpt, step=train_steps, overwrite=True, keep=1)
    else:
        restored_checkpoint = checkpoints.restore_checkpoint(checkpoint_path, target=None)
        params = restored_checkpoint["model"]["params"]
        train_state = trainer.train_state_init(model, lr=1e-3, model_kwargs={'x': jax.random.normal(jrandom.PRNGKey(get_random_int()), x0[0].shape), 't': jnp.array([0]), 'x0': x0[0], 'object_fn': 'Heng'}, retrain=True, ckpt_params=params)
        if retrain:
            train_state, train_loss = trainer.train(train_state, sde_3d, sde_solver, sphere_data_generator_X0, retrain_steps, 6)
            plt.plot(train_loss)
            plt.show()
            # save the model
            config = {"dimension": x0[0].shape}
            ckpt = {"model": train_state, "config": config}
            checkpoints.save_checkpoint(retrain_checkpoint_path, ckpt, step=retrain_steps, overwrite=True, keep=1)
    score_fn = lambda x, t, x0: train_state.apply_fn(train_state.params, x, t, x0)

    reverse_sde = Time_Reversed_SDE_2Dmanifold(sde_3d, score_fn, 1.0,0.01)
    reverse_solver = EulerMaruyama.from_sde(reverse_sde, 0.01, 1.0, 3, condition_x=x0[0],debug_mode=False)
    condition_xs,_ = reverse_solver.solve(xT[0], rng_key=jrandom.PRNGKey(get_random_int()))
    # condition_xs = xs
    condition_xs = np.array(condition_xs)
    plot_trajectory_3d(condition_xs, "reverse_trajectory_finite" + "k_alpha=1.6" + "k_sigma=0.4" + "grid_num=10" + "grid_range=[-1,1]" + "landmark_num=" + str(trainer.landmark_num), simplified=False)

    

    # Create a new figure

    ps.init()
    # global frame_idx
    time = 0.0
    total_time = 1.0
    dt = 0.01
    frame_idx = 0

    ps.set_ground_plane_mode("shadow_only") 
    ps.set_ground_plane_height_mode("manual")
    ps.set_ground_plane_height(-0.2)
    ps.set_view_projection_mode("orthographic")
    ps.look_at((2., 2., 2.), (0., 0., 0.))
    def active_animation():
        for x in condition_xs:
            ps_cloud = ps.register_point_cloud("my points", x)

            # ps_mesh.add_scalar_quantity("scalar", xs[:, 0], enabled=True)



    def imgui_callback():
        global time
        global frame_idx
        
        frame_idx = int(time/dt)
        ps_cloud = ps.register_point_cloud("my points", condition_xs[frame_idx])
        axis_length = 2.0
        x_axis = np.array([[0,0,0], [axis_length,0,0]])
        y_axis = np.array([[0,0,0], [0,axis_length,0]])
        z_axis = np.array([[0,0,0], [0,0,axis_length]])

        ps.register_curve_network("x-axis", np.array([[0,0,0], [axis_length,0,0]]), np.array([[0,1]]))
        ps.register_curve_network("y-axis", np.array([[0,0,0], [0,axis_length,0]]), np.array([[0,1]]))
        ps.register_curve_network("z-axis", np.array([[0,0,0], [0,0,axis_length]]), np.array([[0,1]]))

        # Set axis colors
        ps.get_curve_network("x-axis").set_color((1,0,0))  # Red for X
        ps.get_curve_network("y-axis").set_color((0,1,0))  # Green for Y
        ps.get_curve_network("z-axis").set_color((0,0,1))  # Blue for Z

        



        changed, time = psim.SliderFloat("Time", time, v_min=0,v_max=total_time)

        if changed:
            ps.remove_all_structures()
            frame_idx = int(time/dt)
            time = frame_idx*dt
            ps_cloud = ps.register_point_cloud("my points", condition_xs[frame_idx])
            axis_length = 2.0
            x_axis = np.array([[0,0,0], [axis_length,0,0]])
            y_axis = np.array([[0,0,0], [0,axis_length,0]])
            z_axis = np.array([[0,0,0], [0,0,axis_length]])

            ps_cloud_x0 = ps.register_point_cloud("x0", x0[0])
            ps_cloud_x0.set_material("wax")
            ps_cloud_x0.set_radius(0.005)
            ps_cloud_x0.set_color((1.0,0.3,0.3))
            ps_cloud_x0.set_transparency(0.5)

            ps_cloud_xT = ps.register_point_cloud("xT", xT[0])
            ps_cloud_xT.set_material("wax")
            ps_cloud_xT.set_radius(0.005)
            ps_cloud_xT.set_color((0.3,1.0,0.3))
            ps_cloud_xT.set_transparency(0.5)

            ps.register_curve_network("x-axis", np.array([[0,0,0], [axis_length,0,0]]), np.array([[0,1]]))
            ps.register_curve_network("y-axis", np.array([[0,0,0], [0,axis_length,0]]), np.array([[0,1]]))
            ps.register_curve_network("z-axis", np.array([[0,0,0], [0,0,axis_length]]), np.array([[0,1]]))

            # Set axis colors
            ps.get_curve_network("x-axis").set_color((1,0,0))  # Red for X
            ps.get_curve_network("y-axis").set_color((0,1,0))  # Green for Y
            ps.get_curve_network("z-axis").set_color((0,0,1))  # Blue for Z
            plot_trajectory_3d_polyscope(condition_xs, frame_idx, "reverse_trajectory", simplified=False)

    ps.set_user_callback(imgui_callback)
    ps.show()

