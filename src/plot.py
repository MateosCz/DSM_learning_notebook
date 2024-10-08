import matplotlib.pyplot as plt


def plot_trajectory_2d(trajectory, title):
    fig, ax = plt.subplots()
    for i in range(trajectory.shape[1]):
        ax.plot(trajectory[:, i, 0], trajectory[:, i, 1], marker='o', alpha=0.1, markersize=2, color='orange')
    # plot the shape of the starting point and the ending point
    # enclose the starting point with a circle

    ax.plot(trajectory[0, :, 0], trajectory[0, :, 1], marker='o', color='red', alpha=0.5)
    ax.plot(trajectory[-1, :, 0], trajectory[-1, :, 1], marker='o', color='blue', alpha=0.5)

    ax.set_title(title)
    plt.show()

