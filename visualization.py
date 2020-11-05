import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold


def viz_manifold():
    training = np.loadtxt("data/training.txt")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Y, err = manifold.locally_linear_embedding(training[:1000], n_neighbors=6, n_components=2)
    Y = manifold.Isomap(10, 4).fit_transform(training[:3000])

    ax.scatter(Y[:, 0], Y[:, 1])
    # print(Y[1])
    # print(Y[:, 0])
    plt.show()

    # Fixing random state for reproducibility
    # np.random.seed(19680801)
    #
    # def randrange(n, vmin, vmax):
    #     """
    #     Helper function to make an array of random numbers having shape (n, )
    #     with each number distributed Uniform(vmin, vmax).
    #     """
    #     return (vmax - vmin) * np.random.rand(n) + vmin
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # n = 100
    #
    # # For each set of style and range settings, plot n random points in the box
    # # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    # for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    #     xs = randrange(n, 23, 32)
    #     ys = randrange(n, 0, 100)
    #     zs = randrange(n, zlow, zhigh)
    #     ax.scatter(xs, ys, zs, marker=m)
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    #
    # plt.show()


if __name__ == "__main__":
    viz_manifold()
