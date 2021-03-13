import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold


def plot_manifold(samples: int = 2000, neighbors: int = 10):
    y = np.loadtxt("data/training_v3.txt")[:samples]
    fig = plt.figure(figsize=(12, 6))

    # delta_y = data[:, 4:6] - data[:, :2]
    # y = np.hstack((data[:, :4], delta_y))

    # cmap = get_cmap(data)

    isomap_3d = manifold.Isomap(n_neighbors=neighbors, n_components=3)
    isomap_2d = manifold.Isomap(n_neighbors=neighbors, n_components=2)

    manifold_3d = isomap_3d.fit_transform(y)
    manifold_2d = isomap_2d.fit_transform(y)

    ax = fig.add_subplot(121, projection='3d')
    ax.set_title("ISOMAP run with 3 components")
    ax.scatter(manifold_3d[:, 0], manifold_3d[:, 1], manifold_3d[:, 2])#, c=cmap)
    ax.legend(["faster: red\nslower: orange\nhigher: blue\n lower: cyan"])

    ax = fig.add_subplot(122)
    ax.set_title("ISOMAP run with 2 components")
    ax.scatter(manifold_2d[:, 0], manifold_2d[:, 1])#, c=cmap)

    plt.show()


def plot_manifold_vectors(samples: int = 2000):
    data = np.loadtxt("data/training_v3.txt")[:samples]
    data_normed = np.loadtxt("data/training_normed_v3.txt")[:samples]

    fdata = []

    for i in range(samples):
        if data_normed[i][0] < 0 and abs(data_normed[i][1]) < 0.2:
            fdata.append(data[i])

    data = np.array(fdata)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time')
    ax.set_ylabel('Y-Position')
    ax.set_title('Adverb Mappings in Task Parameter Space')

    # delta_y = data[:, 4:6] - data[:, :2]

    # cmap = get_cmap(data)

    q = plt.quiver(data[:, 2], data[:, 3], data[:, 4], data[:, 5], angles='xy', scale_units='xy', scale=4)#, scale=200, width=0.005, minshaft=0.5, units='xy')
    # ax.quiverkey(q, X=1.15, Y=0.5, U=5, label="faster: red\nslower: orange\nhigher: blue\n lower: cyan", labelpos="S")
    plt.show()


def get_cmap(data: np.array) -> np.array:
    density_factor = 1.5
    cmap = []
    for i in data[:, 2:4]:
        if i[0] != 0:
            # faster/slower
            if i[0] > 0:
                # faster is red
                cmap.append((1.0, 0.0, 0.0, i[0] / density_factor))
            else:
                # slower is orange
                cmap.append((1.0, 0.5, 0.0, i[0] / -density_factor))
        else:
            # higher/lower
            if i[1] > 0:
                # higher is blue
                cmap.append((0.0, 0.0, 1.0, i[1] / density_factor))
            elif i[1] < 0:
                # lower is cyan
                cmap.append((0.0, 0.5, 1.0, i[1] / -density_factor))
            else:
                # no modifiers
                cmap.append((0.25, 0.25, 0.25))
    return cmap


def main():
    plot_manifold_vectors()
    # plot_manifold()


if __name__ == "__main__":
    main()
