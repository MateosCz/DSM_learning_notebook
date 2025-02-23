import src.math.geometry as geometry
import jax.numpy as jnp
lagrange_sde_configs = {
    "sde_type": "lagrange",
    "sde_name": "kunita_lagrange_sde",
    "sde_params": {
        "dt": 0.02,
        "k_alpha": [4.0, 3.0, 2.0, 1.0],
        "k_sigma": [1.5, 1.2, 1.0, 0.8],
    }
}

eulerian_sde_configs = {
    "sde_type": "eulerian",
    "sde_name": "kunita_eulerian_sde",
    "sde_params": {
        "dt": 0.02,
        "k_alpha": [1.5, 2.0, 1.0, 0.5],
        "k_sigma": [1.5, 1.2, 1.0, 0.8],
        "grid_size": 64,
        "grid_range": (-3, 3),
    }
}


data_generator_configs = {
    "data_generator_name": "ellipse_data_generator",
    "data_generator_params": {
        "landmark_num": [32, 64, 128],
        "a": [2, 1.2, 0.8, 1],
        "b": [1, 0.8, 1.2, 2],
        "rotation_matrix": [geometry.get_rotation_matrix(0), geometry.get_rotation_matrix(jnp.pi/4), geometry.get_rotation_matrix(jnp.pi/2), geometry.get_rotation_matrix(3*jnp.pi/4)],
        "center": [jnp.array([0, 0]), jnp.array([0, 0]), jnp.array([0, 0]), jnp.array([0, 0])],
    }
}

