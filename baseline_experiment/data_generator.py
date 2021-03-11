import numpy as np
from numpy.random import default_rng
from collections.abc import Callable
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

rng = default_rng()

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')


def ball_launch(tau: np.array) -> np.array:
    time = tau[0]
    y_pos = tau[1]

    x_vel = 10.0/time
    y_vel = 5.0 * time + y_pos / time

    theta = np.array([x_vel, y_vel])
    return theta


def label_ball_launch(tau: np.array, tau_prime: np.array) -> str:
    delta = tau_prime - tau
    time_label = ""
    y_label = ""

    #TODO Go back to known adverb axes, bake into assumptions

    time_diff = delta[0]
    y_diff = delta[1]

    if abs(time_diff) > 0.25:
        if abs(time_diff) >= 1.5:
            t_mod = "far "
        elif abs(time_diff) >= 1:
            t_mod = "much "
        else:
            t_mod = ""

        if time_diff >= 0:
            time_label += t_mod + "slower"
        else:
            time_label += t_mod + "faster"

    if abs(y_diff) > 2.5:
        if abs(y_diff) >= 15:
            y_mod = "far "
        elif abs(y_diff) >= 10:
            y_mod = "much "
        else:
            y_mod = ""

        if y_diff >= 0:
            y_label += y_mod + "higher"
        else:
            y_label += y_mod + "lower"

    if time_label == y_label == "":
        return ""

    if time_label != "" and y_label != "":
        if rng.integers(0, 2):
            return time_label + " and " + y_label
        else:
            return y_label + " and " + time_label

    if time_label != "":
        return time_label

    if y_label != "":
        return y_label


def generate_samples(count: int, skill: Callable, labeler: Callable, task_min: [], task_max: []) -> np.array:
    samples = []
    for i in range(count):
        # sample random tau
        tau = rng.uniform(task_min, task_max)
        # theta = skill(tau)

        tau_prime = rng.uniform(task_min, task_max)
        # theta_prime = skill(tau_prime)

        samples.append(np.concatenate([tau, tau_prime]))

    samples = np.array(samples)
    labels = [labeler(s[0:2], s[2:4]) for s in samples]
    # label_encodings = sbert_model.encode(labels)
    # samples = np.hstack([label_encodings, samples])


    # print(np.shape(samples))
    return samples


def calculate_normalization_values(samples: []) -> (np.array, np.array):
    inputs = [s[-4:] for s in samples]
    inputs = np.array(inputs, dtype=np.float32)
    mu = np.sum(inputs, axis=0) / len(samples)
    sigma2 = np.sum((inputs - mu) ** 2, axis=0) / len(samples)
    return mu, sigma2


def normalize_samples(samples: [], mu: np.array, sigma2: np.array) -> []:
    inputs = [s[-4:] for s in samples]
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
    data_label = "v2"
    samples_train = generate_samples(2000, ball_launch, label_ball_launch, task_min=np.array([1, -15]), task_max=np.array([4, 15]))
    samples_test = generate_samples(200, ball_launch, label_ball_launch, task_min=np.array([1, -15]), task_max=np.array([4, 15]))

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
    main()
