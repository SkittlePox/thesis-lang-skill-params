import random
import numpy as np


def generate_samples(count: int) -> []:
    samples = []
    for i in range(count):
        vx0 = random.uniform(5, 10)
        dy0 = random.uniform(-10, 10)
        vy0 = 50 * 1 / vx0 + dy0 * 1 / 10 * vx0

        # pick a language command vector
        intensity = random.sample([-3, -2, -1, 0, 1, 2, 3], 1)[0]
        if bool(random.getrandbits(1)):
            # apply intensity to vx (faster)
            vx1 = random.gauss((vx0 + intensity), 0.1)
            dy1 = dy0
            vy1 = 50 * 1 / vx1 + dy1 * 1 / 10 * vx1
            c = [intensity, 0]
        else:
            # apply intensity to dy (higher)
            vx1 = vx0
            dy1 = random.gauss((dy0 + intensity), 0.1)
            vy1 = 50 * 1 / vx1 + dy1 * 1 / 10 * vx1
            c = [0, intensity]

        sample = [vx0, vy0, c[0], c[1], vx1, vy1]
        samples.append(sample)
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
    sigma2 = np.sum((inputs - mu)**2, axis=0) / len(samples)
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
    samples_train = generate_samples_uniform(20000)
    samples_test = generate_samples_uniform(2000)

    np.savetxt("data/training.txt", samples_train)
    np.savetxt("data/testing.txt", samples_test)

    mu, sigma2 = calculate_normalization_values(samples_train)

    np.savetxt("data/mu.txt", mu)
    np.savetxt("data/sigma2.txt", sigma2)

    new_samples_train = normalize_samples(samples_train, mu, sigma2)
    new_samples_test = normalize_samples(samples_test, mu, sigma2)

    np.savetxt("data/training_normed.txt", new_samples_train)
    np.savetxt("data/testing_normed.txt", new_samples_test)


if __name__ == "__main__":
    main()
