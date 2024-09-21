import matplotlib.pyplot as plt


def plot_trajectory_2d(trajectory, title):
    fig, ax = plt.subplots()
    for i in range(trajectory.shape[1]):
        ax.plot(trajectory[:, i, 0], trajectory[:, i, 1], marker='o')
    ax.set_title(title)
    plt.show()