import random
import numpy as np
from collections.abc import Callable
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')


def ball_launch(tau: np.array) -> np.array:
    time = tau[0]
    y_pos = tau[1]

    x_vel = 10.0/time
    y_vel = 5.0 * time + y_pos / time

    theta = np.array([x_vel, y_vel])
    return theta


def label_ball_launch(theta: np.array, theta_prime: np.array) -> str:
    delta = theta_prime - theta
    label = ""
    # TODO estimate adverb changes
    if abs(delta[0]) > 0.5:
        pass
    return delta


def generate_samples(count: int, skill: Callable, task_min: [], task_max: [], task_perturb_diff: np.array) -> np.array:
    samples = []
    for i in range(count):
        # sample random tau
        tau = random.uniform(task_min, task_max)
        theta = skill(tau)

        tau_prime = random.uniform(tau - task_perturb_diff, tau + task_perturb_diff)
        tau_prime = np.clip(tau_prime, task_min, task_max)
        theta_prime = skill(tau_prime)

        samples.append(np.concatenate([tau, theta, tau_prime, theta_prime]))
    return np.array(samples)


def label_samples(samples: np.array, score: Callable) -> np.array:
    labels = [label_ball_launch(s[2:4], s[6:8]) for s in samples]
    return labels


def calculate_normalization_values(samples: []) -> (np.array, np.array):
    inputs = [s[:4] for s in samples]
    inputs = np.array(inputs, dtype=np.float32)
    mu = np.sum(inputs, axis=0) / len(samples)
    sigma2 = np.sum((inputs - mu) ** 2, axis=0) / len(samples)
    return mu, sigma2


def normalize_samples(samples: [], mu: np.array, sigma2: np.array) -> []:
    inputs = [s[:4] for s in samples]
    outputs = [s[4:6] for s in samples]
    inputs = np.array(inputs, dtype=np.float32)
    outputs = np.array(outputs, dtype=np.float32)

    inputs -= mu
    inputs /= sigma2
    outputs -= mu[:2]
    outputs /= sigma2[:2]

    new_samples = np.array([np.concatenate((i, o)) for i, o in zip(inputs, outputs)])
    return new_samples


def main():
    data_label = "v1"
    samples_train = generate_samples(20000)
    samples_test = generate_samples(2000)

    np.savetxt(f"data/training_{data_label}.txt", samples_train)
    np.savetxt(f"data/testing_{data_label}.txt", samples_test)

    mu, sigma2 = calculate_normalization_values(samples_train)

    np.savetxt(f"data/mu_{data_label}.txt", mu)
    np.savetxt(f"data/sigma2_{data_label}.txt", sigma2)

    new_samples_train = normalize_samples(samples_train, mu, sigma2)
    new_samples_test = normalize_samples(samples_test, mu, sigma2)

    np.savetxt(f"data/training_normed_{data_label}.txt", new_samples_train)
    np.savetxt(f"data/testing_normed_{data_label}.txt", new_samples_test)


if __name__ == "__main__":
    # main()
    samples = generate_samples(20, ball_launch, task_min=np.array([1, -15]), task_max=np.array([4, 15]),
                               task_perturb_diff=np.array([2, 15]))
    print(samples)
    s = label_samples(samples, label_ball_launch)
    # print(s)
    # print(len(samples))
    # print(len(samples[:]))
    # print(samples[:][:, [2, 3, 6, 7]])
    # print(samples[0])
    # s = sbert_model.encode(["much higher"])
    print(s)