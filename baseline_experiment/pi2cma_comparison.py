import numpy as np
from data_generator import ball_launch
import pickle
from sklearn.preprocessing import StandardScaler
from net import Net
from data_generator import label_ball_launch_nonlinear
import torch


def pi2cma(weights, rewards, previous_weights, h=10):
    '''
	Path Integral Policy Improvement with Covariance Matrix Adaptation.
	Taken from github.com/babbatem/skills_kin
	From Freek Stulp et. al ICML 2012, Table 1, right side.
	:param weights: weights of policy (n_trials, n_weights)
	:param rewards: instantaneous rewards during rollouts (n_trials, n_steps)
	:param previous_weights: previous mean of parameters (n_weights,)
	:param h: a free parameter. apparently 10 is the correct value from the paper...
	:return: new mean (theta_new) and variance (sigma_new)
	'''
    # params
    n_trials = len(weights)
    n_weights = weights.shape[1]
    max_timesteps = max([len(r) for r in rewards])

    # init placeholders.
    theta_new = np.zeros((max_timesteps, weights.shape[1]))
    sigma_new = np.zeros((max_timesteps, weights.shape[1], weights.shape[1]))

    # scores
    S = np.zeros((max_timesteps, n_trials))

    # probability weights for each episode
    P = np.zeros((max_timesteps, n_trials))

    # loop over time and trials
    for i in range(max_timesteps):
        for k in range(n_trials):

            # make sure we're not out of bounds
            if i < len(rewards[k]):
                # compute score as sum of future rewards (cost)
                S[i, k] = np.sum(rewards[k][i:])  # gamma?

        # compute min and max cost to go for time step i
        min_Si = np.min(S[i, :])
        max_Si = np.max(S[i, :])

        # compute probabilities. note: -1.0*h if you want to minimize cost.
        weighted_S = np.exp(h * (S[i, :] - min_Si) / (max_Si - min_Si))
        P[i, :] = weighted_S / np.sum(weighted_S)

        # parameter updates
        theta_new[i] = np.dot((P[i, :]), weights)
        for k in range(n_trials):
            diff = weights[k] - previous_weights
            diff = diff.reshape(-1, 1)
            prod = np.matmul(diff, diff.T)
            sigma_new[i] += P[i, k] * prod

    # compute weighted temporal average
    N = max_timesteps
    time_vec = N - np.arange(1, N + 1)
    out_theta_new = np.dot(time_vec, theta_new) / np.sum(time_vec)
    out_sigma_new = np.dot(time_vec, sigma_new.reshape(N, -1)) / np.sum(time_vec)
    out_sigma_new = out_sigma_new.reshape(n_weights, n_weights)
    return out_theta_new, out_sigma_new


def calculate_reward(tau: [], goal: [], scaler: StandardScaler):
    goal = np.pad(goal, 2)
    tau_padded = np.pad(tau, 2)
    p = scaler.transform([goal]) - scaler.transform([tau_padded])
    s = -(np.square(p[0][2]) + np.square(p[0][3]))
    return s


def calculate_rewards(tau: [], goal: [], scaler: StandardScaler):
    return [calculate_reward(t, goal, scaler) for t in tau]


def main():
    scaler = pickle.load(open("data/scaler_v8_standard.p", 'rb'))
    goal = np.random.uniform([1, -15], [4, 15])
    print(f"Goal: {goal}")
    start = np.random.uniform([1, -15], [4, 15])
    print(f"Starting Tau: {start}")

    mu = start
    sig2 = np.array([0.75, 75])
    for i in range(50):
        if sig2.shape == (2,):
            samples = np.random.normal(mu, sig2, size=(10, 2))
        else:
            samples = np.random.multivariate_normal(mu, sig2, size=(10,))
        rewards = [[0, r] for r in calculate_rewards(samples, goal, scaler)]
        weights_mean = samples.mean(axis=0)
        mu, sig2 = pi2cma(samples, rewards, weights_mean)
        print(mu)
        reward = calculate_reward(mu, goal, scaler)
        print(reward)
        label = label_ball_launch_nonlinear(mu, (goal - mu))
        print(label)

        if label[0] == 0 and label[1] == 0 or reward > -0.1:
            print(f"Converged after {(i)*10} samples")
            break
        elif i == 19:
            print("Did not converge")
        # d = abs((mu - [2.5, 0.0]) / [3, 30])
        # print(d[0] - d[1])

    net = Net()
    net.load_state_dict(torch.load("data/model.pt"))
    net.eval()

    mu = start
    # mu_normed = scaler.transform([np.pad(mu, 2)])[:, 2:4][0]
    # print(mu_normed)
    for i in range(10):
        label = label_ball_launch_nonlinear(mu, (goal - mu))
        inp = np.hstack([label, mu])
        inp_normed = np.array(scaler.transform([[*inp, 0, 0]])[0][:4], dtype=np.float32)
        p = net.forward(torch.from_numpy(inp_normed))
        p_n = p.detach().numpy()
        # print(p_n)
        print(f"Label: {label}")
        update = scaler.inverse_transform([[0, 0, 0, 0, *p_n]])[0][4:]
        # print(update)
        if label[0] == 0 and label[1] == 0:
            print(f"Bottom out after {i+1} instructions")
            break
        mu = mu + update
        reward = calculate_reward(mu, goal, scaler)
        print(mu)
        print(reward)
        # d = abs((mu - goal) / [3, 30])
        # print(d)

        if reward > -0.1:
            print(f"Converged after {i+1} instructions")
            break


if __name__ == '__main__':
    main()
