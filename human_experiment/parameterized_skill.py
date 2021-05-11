import numpy as np
from sklearn import manifold, svm
import pickle
import matplotlib.pyplot as plt


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SkillNet(nn.Module):
    def __init__(self):
        super(SkillNet, self).__init__()
        self.dense1 = nn.Linear(in_features=3, out_features=30)
        self.dense2 = nn.Linear(in_features=30, out_features=30)
        self.dense3 = nn.Linear(in_features=30, out_features=45)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = F.relu(x)
        x = self.dense3(x)
        return x


def test(net: SkillNet):
    samples = np.loadtxt("data/test_data_normed_v0.txt", dtype=np.float32)
    inputs = torch.tensor(samples[:, :3], requires_grad=False)
    outputs = torch.tensor(samples[:, 3:], requires_grad=False)

    criterion = nn.MSELoss()

    predictions = net.forward(inputs)
    loss = criterion(predictions, outputs)
    print(f"Testing Loss: {loss}")
    return loss


def train(net: SkillNet):
    samples = np.loadtxt("data/train_data_normed_v0.txt", dtype=np.float32)[:800]
    inputs = torch.tensor(samples[:, :3], requires_grad=True)
    outputs = torch.tensor(samples[:, 3:], requires_grad=True)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.2)

    training_loss = []
    testing_loss = []

    for _ in range(400):
        net.zero_grad()
        optimizer.zero_grad()
        predictions = net.forward(inputs)
        loss = criterion(predictions, outputs)
        loss.backward()
        optimizer.step()
        training_loss.append(loss.detach().numpy())
        print(f"Training Loss: {loss}")
        testing_loss.append(test(net).detach().numpy())

    # print(type(training_loss[0]))

    return np.array(training_loss), np.array(testing_loss)


def denormalize(predictions: torch.Tensor, mu: np.array, sigma2: np.array) -> np.array:
    predictions = predictions.detach().numpy()
    predictions *= sigma2[:-2]
    predictions += mu[:-2]
    return predictions


def normalize(samples: np.array, mu: np.array, sigma2: np.array) -> np.array:
    samples -= mu
    samples /= sigma2
    return torch.tensor(samples)


def plot_loss(training_loss: np.array, testing_loss: np.array):
    print(len(training_loss))
    plt.plot(list(range(len(testing_loss))), testing_loss, label='Testing Loss')
    plt.plot(list(range(len(training_loss))), training_loss, label='Training Loss')
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.title("Training and Testing Loss (s=30)")
    plt.legend()
    plt.show()


def svm_predict(svm_models, inputs):
    # preds = []
    # for i in range(len(inputs)):
    #     pred = []
    #     for j in range(inputs[i]):
    #         pred.append(svm_models[j].predict())
    preds = []
    for m in svm_models:
        preds.append(m.predict(inputs))

    return np.array(preds).transpose()


def do_svm():
    samples = np.loadtxt("data/train_data_normed_v0.txt", dtype=np.float32)[:800]
    inputs = samples[:, :3]
    outputs = samples[:, 3:]

    svms = []

    for i in range(np.shape(outputs)[1]):
        clf = svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        clf.fit(inputs, outputs[:, i])
        svms.append(clf)

    # print(inputs[0])
    # print(outputs[0])

    # print(svms[0].predict(inputs[:1]))
    # print(svms[1].predict(inputs[:2]))

    preds = svm_predict(svms, inputs[:5])
    print(preds[:1])
    print(inputs[0])
    print(outputs[0])

    np.savetxt('data/params_test_1', preds[2])


def main():
    net = SkillNet()
    test(net)
    training_loss, testing_loss = train(net)
    torch.save(net.state_dict(), "data/model.pt")
    test(net)
    plot_loss(training_loss, testing_loss)

    # do_svm()

    model_test()


def model_test():
    net = SkillNet()
    net.load_state_dict(torch.load("data/model.pt"))
    net.eval()

    scl = open('data/scaler_v0.p', 'rb')
    scaler = pickle.load(scl)

    data = np.loadtxt('data/train_data_normed_v0.txt', dtype=np.float32)

    inputs = torch.tensor(data[:1, :3])
    params = net.forward(inputs).detach().numpy()[0]
    # params = np.pad(params, (3, 0))
    # params = scaler.inverse_transform(params)[3:]
    np.savetxt('data/params_test_2', params)



def repl():
    net = SkillNet()
    net.load_state_dict(torch.load("data/model.pt"))
    net.eval()

    mu = np.loadtxt("data/mu.txt")
    sigma2 = np.loadtxt("data/sigma2.txt")

    while True:
        s = input("input: ").split()
        arr = np.array(s, dtype=np.float32)
        arr_normed = normalize(arr, mu, sigma2)
        pred = net.forward(arr_normed)
        denormalize(pred, mu, sigma2)
        print(pred)


def isomap():
    fl = open('data/goals_1', 'rb')
    goals = pickle.load(fl)

    fel = open('data/params_1', 'rb')
    params = pickle.load(fel)

    print(np.shape(params))

    X_iso = manifold.Isomap(n_neighbors=30, n_components=2).fit_transform(params)

    plt.scatter(X_iso[:, 0], X_iso[:, 1])

    plt.show()


if __name__ == '__main__':
    main()