import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import matplotlib.collections as mcoll

def plot_trajectory_2d(trajectory, title, simplified=True):
    # trajectory: (time_steps, landmark_num, 2)
    fig, ax = plt.subplots()
    color_range = jnp.linspace(0, 1, trajectory.shape[0])
    # adjust the point size by the landmark number
    point_size = 6 + 2 * jnp.arange(trajectory.shape[1])
    # color_range = jnp.flip(color_range)
    if not simplified:
        for i in range(trajectory.shape[1]):  # iterate over landmark number dimension
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

        start_point_x0 = jnp.array([trajectory[0, 0, 0], trajectory[0, 0, 1]])
        end_point_x0 = jnp.array([trajectory[0, -1, 0], trajectory[0, -1, 1]])
        start_point_xT = jnp.array([trajectory[-1, 0, 0], trajectory[-1, 0, 1]])
        end_point_xT = jnp.array([trajectory[-1, -1, 0], trajectory[-1, -1, 1]])
        envelope_x0 = jnp.array([start_point_x0, end_point_x0])
        envelope_xT = jnp.array([start_point_xT, end_point_xT])
        ax.plot(envelope_x0[:, 0], envelope_x0[:, 1], '-', color=plt.cm.coolwarm(color_range[0]), alpha=0.7)
        ax.plot(envelope_xT[:, 0], envelope_xT[:, 1], '-', color=plt.cm.coolwarm(color_range[-1]), alpha=0.7)
        
        # add color bar
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.coolwarm), ax=ax, orientation='vertical')
        cbar.set_label('Time')
    else:
        for i in range(trajectory.shape[1]):  # iterate over landmark number dimension
            ax.plot(trajectory[:,i,0], trajectory[:,i,1], '-', color='orange', alpha=0.5)
        # plot the shape of the starting point and the ending point and connect them
        ax.plot(trajectory[0, :, 0], trajectory[0, :, 1], 'o', color=plt.cm.coolwarm(color_range[0]), alpha=0.7, markersize=6)
        ax.plot(trajectory[-1, :, 0], trajectory[-1, :, 1], 'o', color=plt.cm.coolwarm(color_range[-1]), alpha=0.7, markersize=6)
        ax.plot(trajectory[0, :, 0], trajectory[0, :, 1], '-', color=plt.cm.coolwarm(color_range[0]), alpha=0.7)# connect the landmarks at the start time
        ax.plot(trajectory[-1, :, 0], trajectory[-1, :, 1], '-', color=plt.cm.coolwarm(color_range[-1]), alpha=0.7)# connect the landmarks at the end time
        # envelope the start and end points
        start_point_x0 = jnp.array([trajectory[0, 0, 0], trajectory[0, 0, 1]])
        end_point_x0 = jnp.array([trajectory[0, -1, 0], trajectory[0, -1, 1]])
        start_point_xT = jnp.array([trajectory[-1, 0, 0], trajectory[-1, 0, 1]])
        end_point_xT = jnp.array([trajectory[-1, -1, 0], trajectory[-1, -1, 1]])
        envelope_x0 = jnp.array([start_point_x0, end_point_x0])
        envelope_xT = jnp.array([start_point_xT, end_point_xT])
        ax.plot(envelope_x0[:, 0], envelope_x0[:, 1], '-', color=plt.cm.coolwarm(color_range[0]), alpha=0.7)
        ax.plot(envelope_xT[:, 0], envelope_xT[:, 1], '-', color=plt.cm.coolwarm(color_range[-1]), alpha=0.7)


    
    # keep the scale of x and y the same
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    plt.show()

#plot the score for a group of landmarks, use quiver to plot the vector field
def plot_score_field(score, t, x0, 
                     xmin=-2.0, xmax=2.0, num_x=20,
                     ymin=-2.0, ymax=2.0, num_y=20,
                     landmark_to_plot=0):
    """
    Plot a score field for a given score function where the score function is defined as:
        score(x, t, x0)
    with:
        x: (landmark_num, 2) array of landmarks coordinates
        t: scalar
        x0: (landmark_num, 2) fixed reference landmarks

    Parameters
    ----------
    score : function
        A function score(x, t, x0) -> (landmark_num, 2)
    t : scalar
        A scalar parameter for the score function.
    x0 : jnp.ndarray, shape (landmark_num, 2)
        Reference landmark positions.
    xmin, xmax, ymin, ymax : float
        Defines the 2D region over which we plot the score field.
    num_x, num_y : int
        Number of points along x and y directions to form the grid.
    landmark_to_plot : int
        Index of the landmark in the output to visualize.

    Returns
    -------
    None
        Displays a quiver plot of the selected landmark's score field.
    """

    # Generate a grid of points
    xs = jnp.linspace(xmin, xmax, num_x)
    ys = jnp.linspace(ymin, ymax, num_y)
    X, Y = jnp.meshgrid(xs, ys)

    # Flatten the grid into (num_points, 2)
    points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)  # (num_points, 2)

    # Function to create an input x array for each point
    # We vary the position of the chosen landmark along the grid,
    # while keeping other landmarks the same as x0.
    def make_x_for_point(x_val, y_val):
        modified_x = x0.at[landmark_to_plot, 0].set(x_val)
        modified_x = modified_x.at[landmark_to_plot, 1].set(y_val)
        return modified_x

    create_x_array_for_point = jax.vmap(make_x_for_point, in_axes=(0,0))
    X_array = create_x_array_for_point(points[:,0], points[:,1])
    # X_array: (num_points, landmark_num, 2)

    # Vectorized score evaluation
    def batch_score(x):
        return score(x, t, x0)  # shape: (landmark_num, 2)

    S = jax.vmap(batch_score, in_axes=(0,))(X_array) 
    # S: (num_points, landmark_num, 2)

    # Extract the vector field for the chosen landmark
    U = S[:, landmark_to_plot, 0]
    V = S[:, landmark_to_plot, 1]

    # Reshape to (num_y, num_x)
    U = U.reshape(X.shape)
    V = V.reshape(Y.shape)

    # Plot using quiver
    plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=20, color='red', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Score Field at t={t}')
    plt.axis('equal')
    plt.show()


#plot the matrix and show the heatmap with the color bar
def plot_matrix(matrix, title):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='coolwarm')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.show()