import random
import numpy as np


def skill(tau: []) -> []:
    x_vel = tau[0]
    y_pos = tau[1]

    y_vel = 50 * 1 / x_vel + y_pos * 1 / 10 * x_vel

    theta = [x_vel, y_vel]
    return theta


def generate_samples(count: int) -> []:
    samples = []
    for i in range(count):
        # sample random tau
        x_vel = random.uniform(5, 10)
        y_pos = random.uniform(-10, 10)
        tau = np.array([x_vel, y_pos])

        # pick a lambda and its corresponding delta_tau
        delta = random.sample([-3. -2, -1, 0, 1, 2, 3], 1)[0]
        if bool(random.getrandbits(1)):
            # apply delta to x velocity
            x_vel2 = random.gauss((x_vel + delta), 0.1) # for fuzziness
            y_pos2 = y_pos
            lam = np.array([delta, 0])
        else:
            # apply delta to y position
            x_vel2 = x_vel
            y_pos2 = random.gauss((y_pos + delta), 0.1)
            lam = np.array([0, delta])
        tau2 = np.array([x_vel2, y_pos2])
        delta_tau = tau2 - tau
        samples.append(np.concatenate([tau, lam, delta_tau]))
    return samples


def generate_samples_uniform(count: int) -> []:
    samples = []

    for i in range(count):
        vx0 = random.uniform(5, 10)
        dy0 = random.uniform(-10, 10)
        vy0 = 50 * 1 / vx0 + dy0 * 1 / 10 * vx0

        if bool(random.getrandbits(1)):
            # apply intensity to vx (faster)
            vx1 = random.uniform(vx0 - 2.5, vx0 + 2.5)
            dy1 = dy0
            vy1 = 50 * 1 / vx1 + dy1 * 1 / 10 * vx1
            if vx1 > vx0:
                intensity = 1
            else:
                intensity = -1
            c = [intensity, 0]
        else:
            # apply intensity to dy (higher)
            vx1 = vx0
            dy1 = random.uniform(dy0 - 10, dy0 + 10)
            vy1 = 50 * 1 / vx1 + dy1 * 1 / 10 * vx1

            if dy1 > dy0:
                intensity = 1
            else:
                intensity = -1
            c = [0, intensity]

        sample = [vx0, vy0, c[0], c[1], vx1, vy1]
        samples.append(sample)
    return samples


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
    data_label = "v0"
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
    main()