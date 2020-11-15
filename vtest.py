import matplotlib.pyplot as plt
import numpy as np


def plot_manifold_vectors(samples: int = 5000):
    data = np.loadtxt("data/training.txt")[:samples]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Vx')
    ax.set_ylabel('Vy')
    ax.set_title('Parameter Manifold')

    delta_y = data[:, 4:6] - data[:, :2]

    adv_map = []
    for i in data[:, 2:4]:
        if i[0] != 0:
            # faster/slower
            if i[0] > 0:
                # faster is red
                adv_map.append((1.0, 0.0, 0.0, i[0] / 3))
            else:
                # slower is orange
                adv_map.append((1.0, 0.5, 0.0, i[0] / -3))
        else:
            # higher/lower
            if i[1] > 0:
                # higher is blue
                adv_map.append((0.0, 0.0, 1.0, i[1] / 3))
            else:
                # lower is cyan
                adv_map.append((0.0, 0.5, 1.0, i[1] / -3))

    q = plt.quiver(data[:, 0], data[:, 1], delta_y[:, 0], delta_y[:, 1], color=adv_map, scale=100, width=0.002, minshaft=0.5)
    ax.quiverkey(q, X=1.15, Y=0.5, U=5, label="faster: red\nslower: orange\nhigher: blue\n lower: cyan", labelpos="S")
    plt.show()


def main():
    plot_manifold_vectors()


if __name__ == "__main__":
    main()
