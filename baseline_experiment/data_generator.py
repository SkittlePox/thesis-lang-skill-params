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


def label_ball_launch_nonlinear(tau: np.array, delta_tau: np.array):
    time_label = ""
    y_label = ""

    t_val = 0.0
    y_val = 0.0

    time = (-1 * tau[0]) + 4
    y = tau[1] + 15

    time_diff = delta_tau[0]
    y_diff = delta_tau[1]

    if abs(time_diff) > 0.05 + time * 0.15:
        if abs(time_diff) >= 1.2 + time * 0.17:
            t_mod = "far "
            t_val = 3.0
        elif abs(time_diff) >= 0.6 + time * 0.15:
            t_mod = "much "
            t_val = 2.0
        else:
            t_mod = ""
            t_val = 1.0

        if time_diff >= 0:
            time_label += t_mod + "slower"
            t_val *= -1.0
        else:
            time_label += t_mod + "faster"

    if abs(y_diff) > 0.5 + y * 0.15:
        if abs(y_diff) >= 12 + y * 0.17:
            y_mod = "far "
            y_val = 3.0
        elif abs(y_diff) >= 6 + y * 0.15:
            y_mod = "much "
            y_val = 2.0
        else:
            y_mod = ""
            y_val = 1.0

        if y_diff >= 0:
            y_label += y_mod + "higher"
        else:
            y_label += y_mod + "lower"
            y_val *= -1.0

    return np.array([t_val, y_val])


def label_ball_launch(tau: np.array, delta_tau: np.array):
    time_label = ""
    y_label = ""

    t_val = 0.0
    y_val = 0.0

    time_diff = delta_tau[0]
    y_diff = delta_tau[1]

    if abs(time_diff) > 0.25:
        if abs(time_diff) >= 1.5:
            t_mod = "far "
            t_val = 3.0
        elif abs(time_diff) >= 1:
            t_mod = "much "
            t_val = 2.0
        else:
            t_mod = ""
            t_val = 1.0

        if time_diff >= 0:
            time_label += t_mod + "slower"
            t_val *= -1.0
        else:
            time_label += t_mod + "faster"

    if abs(y_diff) > 2.5:
        if abs(y_diff) >= 15:
            y_mod = "far "
            y_val = 3.0
        elif abs(y_diff) >= 10:
            y_mod = "much "
            y_val = 2.0
        else:
            y_mod = ""
            y_val = 1.0

        if y_diff >= 0:
            y_label += y_mod + "higher"
        else:
            y_label += y_mod + "lower"
            y_val *= -1.0

    return np.array([t_val, y_val])

    # if time_label == y_label == "":
    #     return ""
    #
    # if time_label != "" and y_label != "":
    #     if rng.integers(0, 2):
    #         return time_label + " and " + y_label
    #     else:
    #         return y_label + " and " + time_label
    #
    # if time_label != "":
    #     return time_label
    #
    # if y_label != "":
    #     return y_label


def generate_samples(count: int, skill: Callable, labeler: Callable, task_min: [], task_max: []) -> np.array:
    samples = []
    for i in range(count):
        # sample random tau
        tau = rng.uniform(task_min, task_max)
        # theta = skill(tau)

        tau_prime = rng.uniform(task_min, task_max)

        delta_tau = tau_prime - tau
        # theta_prime = skill(tau_prime)

        samples.append(np.concatenate([tau, delta_tau]))

    samples = np.array(samples)
    labels = np.array([labeler(s[0:2], s[2:4]) for s in samples])
    # label_encodings = sbert_model.encode(labels)
    # samples = np.hstack([label_encodings, samples])


    # print(np.shape(samples))
    # print(labels)
    samples = np.hstack([labels, samples])
    return samples


def calculate_normalization_values(samples: []) -> (np.array, np.array):
    inputs = np.array(samples, dtype=np.float32)
    mu = np.sum(inputs, axis=0) / len(samples)
    sigma2 = np.sum((inputs - mu) ** 2, axis=0) / len(samples)
    return mu, sigma2


def normalize_samples(samples: [], mu: np.array, sigma2: np.array) -> []:
    # inputs = [s[-4:] for s in samples]
    # outputs = [s[4:6] for s in samples]
    # inputs = np.array(inputs, dtype=np.float32)
    # outputs = np.array(outputs, dtype=np.float32)
    samples = np.array(samples, dtype=np.float32)
    samples -= mu
    samples /= sigma2

    return samples

    # inputs -= mu
    # inputs /= sigma2
    # outputs -= mu[:2]
    # outputs /= sigma2[:2]
    #
    # new_samples = np.array([np.concatenate((i, o)) for i, o in zip(inputs, outputs)])
    # return new_samples

def main():
    data_label = "v5"
    samples_train = generate_samples(500, ball_launch, label_ball_launch_nonlinear, task_min=np.array([1, -15]), task_max=np.array([4, 15]))
    samples_test = generate_samples(2000, ball_launch, label_ball_launch_nonlinear, task_min=np.array([1, -15]), task_max=np.array([4, 15]))

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
    # s = generate_samples(500, ball_launch, label_ball_launch, task_min=np.array([1, -15]), task_max=np.array([4, 15]))
    # mu, sigma2 = calculate_normalization_values(s)
    # print(mu)
    # print(sigma2)
    # ss = normalize_samples(s, mu, sigma2)
    # print(ss)