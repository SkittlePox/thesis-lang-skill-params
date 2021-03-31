import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
from sklearn.preprocessing import normalize


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


def plot_embeddings(samples: int):
    data = np.loadtxt("data/testing_v9_standard.txt")[:samples]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Faster              Slower')
    ax.set_ylabel('Lower              Higher')
    ax.set_title('Distribution of Adverb Labelling (s=2000)')

    heatmap = np.zeros((7,7))



    for d in data:
        heatmap[int(d[1])+3][int(d[0])+3] += 1
        # heatmap[0][int(d[0]) + 3] += 1
        # heatmap[int(d[1])+3][0] += 1

    # heatmap = normalize(heatmap, norm='l1', axis=0)
    heatmap = heatmap/samples
    trunc_heatmap = np.around(heatmap, decimals=3)

    labels = ["-3", "-2", "-1", "0", "1", "2", "3"]

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    # ax.scatter(data[:, 0], data[:, 1])

    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, trunc_heatmap[i, j],
                           ha="center", va="center", color="w")

    ax.set_xticklabels(labels)
    labels.reverse()
    ax.set_yticklabels(labels)
    im = ax.imshow(heatmap)
    plt.show()


def plot_reward_contour():
    net_steps = np.array([[  1.44922374, -13.95353649],[  3.37205507,   4.9808529 ],[  3.73233298,  14.7919342 ]])
    pi2_steps = np.array([[ 3.73233298, 14.7919342 ],
 [ 3.47278675, 15.21327113]])
    pure_steps = np.array([[  1.44922374, -13.95353649],
 [  2.85249509,  -3.92525738],
 [  3.78457802,   4.53001243],
 [  3.93570012,   7.83859085],
 [  3.76901738,  10.96108929],
 [  3.83931687,  12.86365042],
 [  3.68409637,  14.34141549],
[  3.4,  14.5]]
)

    def step_to_arrows(arr):
        arrows = []
        for i in range(1, len(arr)):
            arrows.append(arr[i]-arr[i-1])
        return np.array(arrows)

    net_arrows = step_to_arrows(net_steps)
    pi2_arrows = step_to_arrows(pi2_steps)
    pure_arrows = step_to_arrows(pure_steps)

    goal = [3.5, 15]
    delta = 0.025
    x = np.arange(1.0, 5.0, delta)
    y = np.arange(-15.0, 17.0, delta)
    X, Y = np.meshgrid(x, y)
    Z = (np.square(goal[0]-X) + np.square((goal[1]-Y)/7))
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, levels=20)
    CS2 = ax.contour(X, Y, Z, levels=[0.025], colors='red', alpha=0.6)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.scatter(1.44922374, -13.95353649)
    ax.scatter(3.5, 15, color='red', alpha=0.6)


    plt.quiver(net_steps[:-1, 0], net_steps[:-1, 1], net_arrows[:, 0],
               net_arrows[:, 1], scale=1, scale_units='xy', angles='xy',
               alpha=0.75, color='green', width=0.01,
               headwidth=2.5, headaxislength=4, headlength=4.5,
               label='Adverb-Skill Grounding')
    plt.quiver(pi2_steps[:-1, 0], pi2_steps[:-1, 1], pi2_arrows[:, 0],
               pi2_arrows[:, 1], scale=1, scale_units='xy', angles='xy',
               alpha=0.75, color='k', width=0.01,
               headwidth=2.5, headaxislength=4, headlength=4.5)
    plt.quiver(pure_steps[:-1, 0], pure_steps[:-1, 1], pure_arrows[:, 0],
               pure_arrows[:, 1], scale=1, scale_units='xy', angles='xy',
               alpha=0.7, color='k', width=0.01,
               headwidth=2.5, headaxislength=4, headlength=4.5,
               label='PI2-CMA')


    ax.legend()
    ax.set_title('Paths from [1.45, -14] to [3.5, 15]')
    ax.set_xlabel('Time Parameter')
    ax.set_ylabel('Y-Position Parameter')
    plt.show()


def main():
    # plot_manifold_vectors()
    # plot_manifold()
    # plot_embeddings(2000)
    plot_reward_contour()

if __name__ == "__main__":
    main()
