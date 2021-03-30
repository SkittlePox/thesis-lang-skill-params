import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(in_features=4, out_features=25)
        self.dense2 = nn.Linear(in_features=25, out_features=10)
        self.dense3 = nn.Linear(in_features=10, out_features=2)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = F.relu(x)
        x = self.dense3(x)
        return x


def test(net: Net):
    samples = np.loadtxt("data/testing_normed_v9_standard.txt", dtype=np.float32)
    inputs = torch.tensor(samples[:, :4], requires_grad=False)
    outputs = torch.tensor(samples[:, -2:], requires_grad=False)

    criterion = nn.MSELoss()

    predictions = net.forward(inputs)
    loss = criterion(predictions, outputs)
    print(f"Testing Loss: {loss}")
    return loss


def train(net: Net):
    samples = np.loadtxt("data/training_normed_v9_standard.txt", dtype=np.float32)[:50]
    inputs = torch.tensor(samples[:, :4], requires_grad=True)
    outputs = torch.tensor(samples[:, -2:], requires_grad=True)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    training_loss = []
    testing_loss = []

    for _ in range(200):
        optimizer.zero_grad()
        predictions = net.forward(inputs)
        loss = criterion(predictions, outputs)
        loss.backward()
        optimizer.step()
        training_loss.append(loss)
        print(f"Training Loss: {loss}")
        testing_loss.append(test(net))

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


def main():
    net = Net()
    test(net)
    training_loss, testing_loss = train(net)
    torch.save(net.state_dict(), "data/model.pt")
    test(net)
    plot_loss(training_loss, testing_loss)


def repl():
    net = Net()
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


if __name__ == "__main__":
    main()
    # repl()
