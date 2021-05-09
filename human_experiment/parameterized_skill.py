import numpy as np
from sklearn import manifold
import pickle
import matplotlib.pyplot as plt


def main():
    fl = open('data/goals_1', 'rb')
    goals = pickle.load(fl)

    fel = open('data/params_1', 'rb')
    params = pickle.load(fel)

    print(np.shape(params))

    # print(np.shape(goals))
    # print(np.shape(params))

    # datapoints = np.concatenate((goals, params), axis=1)

    # print(np.shape(datapoints))

    X_iso = manifold.Isomap(n_neighbors=3, n_components=2).fit_transform(params)

    plt.scatter(X_iso[:, 0], X_iso[:, 1])

    plt.show()

if __name__ == '__main__':
    main()