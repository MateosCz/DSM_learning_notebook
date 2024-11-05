import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import matplotlib.collections as mcoll

def plot_trajectory_2d(trajectory, title, simplified=True):
    fig, ax = plt.subplots()
    color_range = jnp.linspace(0, 1, trajectory.shape[0])
    # color_range = jnp.flip(color_range)
    if not simplified:
        for i in range(trajectory.shape[1]):  # iterate over batch dimension
            x = trajectory[:,i,0]
            y = trajectory[:,i,1]
            # Reshape segments to be pairs of points
            points = jnp.stack([x, y], axis=1)  # shape: (time_steps, 2)
            segments = jnp.stack([points[:-1], points[1:]], axis=1)  # shape: (time_steps-1, 2, 2)
        
            lc = mcoll.LineCollection(segments, cmap=plt.cm.coolwarm, alpha=0.8)
            lc.set_array(color_range[:-1])  # one less than points due to segments
            ax.add_collection(lc)
        # plot the shape of the starting point and the ending point and connect them
        ax.plot(trajectory[0, :, 0], trajectory[0, :, 1], 'o', color=plt.cm.coolwarm(color_range[0]), alpha=0.9, markersize=6)
        ax.plot(trajectory[-1, :, 0], trajectory[-1, :, 1], 'o', color=plt.cm.coolwarm(color_range[-1]), alpha=0.9, markersize=6)
        ax.plot(trajectory[0, :, 0], trajectory[0, :, 1], '-', color=plt.cm.coolwarm(color_range[0]), alpha=0.9)
        ax.plot(trajectory[-1, :, 0], trajectory[-1, :, 1], '-', color=plt.cm.coolwarm(color_range[-1]), alpha=0.9)
        
        # add color bar
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.coolwarm), ax=ax, orientation='vertical')
        cbar.set_label('Time')
    else:
        for i in range(trajectory.shape[1]):  # iterate over batch dimension
            ax.plot(trajectory[:,i,0], trajectory[:,i,1], '-', color='orange', alpha=0.5)
        # plot the shape of the starting point and the ending point and connect them
        ax.plot(trajectory[0, :, 0], trajectory[0, :, 1], 'o', color=plt.cm.coolwarm(color_range[0]), alpha=0.7, markersize=6)
        ax.plot(trajectory[-1, :, 0], trajectory[-1, :, 1], 'o', color=plt.cm.coolwarm(color_range[-1]), alpha=0.7, markersize=6)
        ax.plot(trajectory[0, :, 0], trajectory[0, :, 1], '-', color=plt.cm.coolwarm(color_range[0]), alpha=0.7)
        ax.plot(trajectory[-1, :, 0], trajectory[-1, :, 1], '-', color=plt.cm.coolwarm(color_range[-1]), alpha=0.7)

    
    
    ax.set_title(title)
    plt.show()

