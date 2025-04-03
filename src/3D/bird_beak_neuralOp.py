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
import src.SDESolver as SDESolver
import random
import polyscope as ps
from src.models import DsmModel
import src.Trainer as Trainer
import matplotlib.pyplot as plt
import pandas as pd
from src.NeuralOp.neural_operator import CTUNO1D, CTUNO2D
from src.data.BirdBeakData import BirdBeakDataGenerator
from src.plot import plot_trajectory_3d, plot_trajectory_3d_polyscope
from flax.training import checkpoints
import orbax.checkpoint

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def get_random_int():
    return random.randint(0, 1000000)

def load_beak_data_np():
    # Load the beak data
    beaks = pd.read_csv(project_root() + "/data/beaks/PROC_LANDMARKS_090125.csv", sep=",", header = 0)
    landmarks = beaks.iloc[:, 1:]
    # scale the landmarks to the range of [-1, 1]
    landmarks = landmarks.apply(lambda x: x * 2)
    xs = landmarks.iloc[:, 0::3]
    ys = landmarks.iloc[:, 1::3]
    zs = landmarks.iloc[:, 2::3]
    xs = xs.to_numpy()
    ys = ys.to_numpy()
    zs = zs.to_numpy()
    landmarks = landmarks.to_numpy()
    landmarks = landmarks.reshape(xs.shape[0], xs.shape[1], 3)
    names = beaks.iloc[:, 0]
    names = names.to_numpy()
    return names, landmarks


def show_coordinates(axis_length: float):
    x_axis = np.array([[0,0,0], [axis_length,0,0]])
    y_axis = np.array([[0,0,0], [0,axis_length,0]])
    z_axis = np.array([[0,0,0], [0,0,axis_length]])
    ps.register_curve_network("x-axis", x_axis, np.array([[0,1]]))
    ps.register_curve_network("y-axis", y_axis, np.array([[0,1]]))
    ps.register_curve_network("z-axis", z_axis, np.array([[0,1]]))
    ps.get_curve_network("x-axis").set_color((1,0,0))
    ps.get_curve_network("y-axis").set_color((0,0,1))
    ps.get_curve_network("z-axis").set_color((0,1,0))


if __name__ == "__main__":
    names, landmarks = load_beak_data_np()
    dt = 0.01
    total_time = 1.0
    dim = 3
    time = 0.0
    landmarks = jnp.array(landmarks)
    x0 = landmarks[1]
    name_x0 = names[1]
    name_xT = names[3]
    print(name_x0)
    print(name_xT)
    data_generator = BirdBeakDataGenerator(x0, seed=get_random_int())
    print(x0.shape)
    x0 = data_generator.generate_data(x0.shape[0], 4)

    # Kunita flow Eularian
    # sde = Kunita_Flow_SDE_3D_Eulerian(k_alpha=10.0, k_sigma=0.075, grid_num=15, grid_range=(-0.5, 0.5), x0=x0[0])
    # sde = Kunita_Lagrange_SDE(sigma=1, kappa=0.3, x0=x0[0])
    sde = Kunita_Flow_SDE_3D_Eulerian_Optimized(k_alpha=8.0, k_sigma=0.2, grid_num=15, grid_range=(-0.5, 0.5), x0=x0[0], batch_size=256)
    sde_solver = SDESolver.EulerMaruyama.from_sde(sde, dt=0.01, total_time=1.0, dim=3)
    trainer = Trainer.SsmTrainer(seed=get_random_int(), landmark_num=x0[0].shape[0])
    model = CTUNO2D(out_co_dim=3, lifting_dim=32, co_dims_fmults=(1, 2, 4, 8), n_modes_per_layer=(48, 32, 16, 8), norm="instance", act="gelu")
    checkpoint_path = project_root() + "/checkpoints/bird_beak_neuralOp_checkpoint_retrained"
    trainer = Trainer.NeuralOpTrainer(seed=get_random_int(), landmark_num=x0[0].shape[0])
    retrain = False
    if os.path.exists(checkpoint_path):
        restored_checkpoint = checkpoints.restore_checkpoint(checkpoint_path, target=None)
        params = restored_checkpoint["model"]["params"]
        train_state = trainer.train_state_init(model, lr=5e-3, model_kwargs={'x': jax.random.normal(jrandom.PRNGKey(get_random_int()), x0[0].shape), 't': jnp.array([0]), 'x0': x0[0], 'object_fn': 'Heng'}, retrain=retrain, ckpt_params=params)
        if retrain:
            train_state, train_loss = trainer.train(train_state, sde, sde_solver, data_generator, 1000, 14)
            plt.plot(train_loss)
            plt.show()
            checkpoint_path = project_root() + "/checkpoints/bird_beak_neuralOp_checkpoint_retrained"
            config = {"dimension": x0[0].shape}
            ckpt = {"model": train_state, "config": config}
            checkpoints.save_checkpoint(checkpoint_path, ckpt, step=3000, overwrite=True, keep=1)


    else:
        train_state = trainer.train_state_init(model, lr=5e-3, model_kwargs={'x': jax.random.normal(jrandom.PRNGKey(get_random_int()), x0[0].shape), 't': jnp.array([0]), 'x0': x0[0], 'object_fn': 'Heng'})
        train_state, train_loss = trainer.train(train_state, sde, sde_solver, data_generator, 1000, 14)
        config = {"dimension": x0[0].shape}
        ckpt = {"model": train_state, "config": config}
        
        checkpoints.save_checkpoint(checkpoint_path, ckpt, step=1000, overwrite=True, keep=1)
        plt.plot(train_loss)
        plt.show()
        params = train_state.params
    xT = landmarks[60]

    xT_data_generator = BirdBeakDataGenerator(xT, seed=get_random_int())
    xT = xT_data_generator.generate_data(xT.shape[0], 4)
    score_fn = lambda x, t, x0: train_state.apply_fn(params, x, t)




    reverse_sde = Time_Reversed_SDE(sde, score_fn, 1.0, 0.01)
    reverse_solver = SDESolver.EulerMaruyama.from_sde(reverse_sde, dt=0.01, total_time=1.0, dim=3, condition_x=x0[0])
    condition_xs,_ = reverse_solver.solve(xT[0], rng_key=jrandom.PRNGKey(get_random_int()))
    plot_trajectory_3d(condition_xs, "reverse_trajectory", simplified=False)
    condition_xs = np.array(condition_xs)
    

    # condition_xs, _ = sde_solver.solve(x0[0], rng_key=jrandom.PRNGKey(get_random_int()))
    # condition_xs = np.array(condition_xs)



    ps.init()
    frame_idx = 0
    ps.set_ground_plane_mode("shadow_only") 
    ps.set_ground_plane_height_mode("manual")
    ps.set_ground_plane_height(-0.2)
    ps.set_view_projection_mode("orthographic")
    ps.look_at((2., 2., 2.), (0., 0., 0.))
    def imgui_callback():
        global frame_idx, time
        changed, time = psim.SliderFloat("Time", time, v_min=0, v_max=total_time)
        frame_idx = int(time / dt)
        time = frame_idx * dt
        ps_cloud_x0 = ps.register_point_cloud(name_x0, x0[0])
        ps_cloud_x0.set_material("wax")
        ps_cloud_x0.set_radius(0.005)
        ps_cloud_x0.set_color((1.0,0.3,0.3))
        ps_cloud_x0.set_transparency(0.5)

        ps_cloud_xT = ps.register_point_cloud(name_xT, xT[0])
        ps_cloud_xT.set_material("wax")
        ps_cloud_xT.set_radius(0.005)
        ps_cloud_xT.set_color((0.3,1.0,0.3))
        ps_cloud_xT.set_transparency(0.5)

        ps_cloud = ps.register_point_cloud("from_x0"+str(name_x0)+"to_xT"+str(name_xT), condition_xs[frame_idx])
        ps_cloud.set_material("wax")
        ps_cloud.set_radius(0.006)
        ps_cloud.set_color((0.3,0.3,0.8))
        
        show_coordinates(1.0)
        
        if changed:

            ps.remove_all_structures()
            ps_cloud_x0 = ps.register_point_cloud(name_x0, x0[0])
            ps_cloud_x0.set_material("wax")
            ps_cloud_x0.set_radius(0.005)
            ps_cloud_x0.set_color((1.0,0.3,0.3))
            ps_cloud_x0.set_transparency(0.5)

            ps_cloud_xT = ps.register_point_cloud(name_xT, xT[0])
            ps_cloud_xT.set_material("wax")
            ps_cloud_xT.set_radius(0.005)
            ps_cloud_xT.set_color((0.3,1.0,0.3))
            ps_cloud_xT.set_transparency(0.5)

            ps_cloud = ps.register_point_cloud("from_x0"+str(name_x0)+"to_xT"+str(name_xT), condition_xs[frame_idx])
            ps_cloud.set_material("wax")
            ps_cloud.set_radius(0.006)
            ps_cloud.set_color((0.3,0.3,0.8))
            plot_trajectory_3d_polyscope(condition_xs, frame_idx, "reverse_trajectory", simplified=False)
            show_coordinates(1.0)
            
    ps.set_user_callback(imgui_callback)
    ps.show()
