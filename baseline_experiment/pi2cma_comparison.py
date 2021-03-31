import numpy as np
from data_generator import ball_launch
import pickle
from sklearn.preprocessing import StandardScaler
from net import Net
from data_generator import label_ball_launch_nonlinear
import torch
from numpy.random import default_rng

rng = default_rng()


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
    p = goal - tau
    s = -(np.square(p[0]) + np.square(p[1]/7))
    return s


def calculate_rewards(tau: [], goal: [], scaler: StandardScaler):
    return [calculate_reward(t, goal, scaler) for t in tau]


def test():
    scaler = pickle.load(open("data/scaler_v8_standard.p", 'rb'))
    # goal = rng.uniform([1, -15], [4, 15])
    goal = np.array([3, 10])

    print(calculate_rewards([[3, 10.5], [3.1, 10], [2.9, 9.5], [3.1, 10.5], [2.9, 10], [3, 9.5], goal], goal, scaler))


def run_pi2cma(mu, sig2, sample_size, goal, scaler, reward_threshold, human_reward_threshold):
    reward = calculate_reward(mu, goal, scaler)
    if reward > human_reward_threshold:
        print("Human Reward reached by Net")
        human_reward_reached = True
    else:
        human_reward_reached = False

    mu_reward_steps = [[mu, reward]]

    for i in range(1000):
        if sig2.shape == (2,):
            samples = rng.normal(mu, sig2, size=(sample_size, 2))
        else:
            samples = rng.multivariate_normal(mu, sig2, size=(sample_size,))
        rewards = [[0, r] for r in calculate_rewards(samples, goal, scaler)]
        weights_mean = samples.mean(axis=0)
        mu, sig2 = pi2cma(samples, rewards, weights_mean)
        # print(mu)
        # print(sig2)
        reward = calculate_reward(mu, goal, scaler)

        mu_reward_steps.append([mu, reward])
        # print(reward)
        # label = label_ball_launch_nonlinear(mu, (goal - mu))
        # print(label)
        # print(mu, reward)

        if reward > human_reward_threshold and not human_reward_reached:
            print(f"PI2CMA Reached Human Reward after {(i+1)*sample_size} samples\n\t{reward}\n\t{mu}")
            human_reward_reached = True
        if reward > reward_threshold:
            print(f"PI2CMA Converged after {(i + 1) * sample_size} samples\n\t{reward}\n\t{mu}")
            return (i + 1) * sample_size, mu_reward_steps
    print(f"PI2CMA Did not converge after {1000 * sample_size} samples\n\t{reward}\n\t{mu}")
    return None, mu_reward_steps


def run_net(mu, sig2, net, sample_size, goal, scaler, reward_threshold, human_reward_threshold):
    reward = calculate_reward(mu, goal, scaler)
    mu_reward_steps = [[mu, reward]]
    pi2_mu_reward_steps = []
    for i in range(20):
        label = label_ball_launch_nonlinear(mu, (goal - mu))
        if label[0] == 0 and label[1] == 0 or reward > human_reward_threshold:
            print(f"Net bottomed out after {i} instructions\n\t{reward}\n\t{mu}\nMoving to PI2CMA")
            # Run PI2CMA
            num_pi_runs, pi2_mu_reward_steps = run_pi2cma(mu, sig2, sample_size, goal, scaler, reward_threshold, human_reward_threshold)
            if num_pi_runs is not None:
                print(f"Instructions + samples: {num_pi_runs + i}")
                return (i, num_pi_runs), (mu_reward_steps, pi2_mu_reward_steps)
            return None, (mu_reward_steps, pi2_mu_reward_steps)
        inp = np.hstack([label, mu])
        inp_normed = np.array(scaler.transform([[*inp, 0, 0]])[0][:4], dtype=np.float32)
        p = net.forward(torch.from_numpy(inp_normed))
        p_n = p.detach().numpy()
        update = scaler.inverse_transform([[0, 0, 0, 0, *p_n]])[0][4:]
        mu = mu + update
        reward = calculate_reward(mu, goal, scaler)
        mu_reward_steps.append([mu, reward])

        if reward > reward_threshold:
            print(f"Net converged after {i+1} instructions\n\t{reward}\n\t{mu}\n\t{label}")
            return (i+1, 0), (mu_reward_steps, pi2_mu_reward_steps)
    return None, (mu_reward_steps, pi2_mu_reward_steps)


def main():
    total_examples = 10

    sample_size = 15
    reward_threshold = -0.05
    human_reward_threshold = -0.5
    initial_goal_threshold = -20
    sig2 = np.array([3, 15])
    sig2_hybrid = np.array([1.5, 7])
    scaler = pickle.load(open("data/scaler_v8_standard.p", 'rb'))

    pure = []
    hybrid = []

    for i in range(total_examples):
        label = [0, 0]
        start = [0, 0]
        goal = [0, 0]
        reward = -1000

        while label[0] == 0 and label[1] == 0 or reward > initial_goal_threshold:
            start = rng.uniform([1, -15], [4, 15])
            goal = rng.uniform([1, -15], [4, 15])
            # goal = np.array([3.5, 15])
            # start = np.array([1.44922374, -13.95353649])
            label = label_ball_launch_nonlinear(start, (goal - start))
            reward = calculate_reward(start, goal, scaler)

        print("----------")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"Initial Reward: {reward}")
        print(f"Epsilon: {reward_threshold}")
        print("----------")

        pure_pi2cma_result, pure_mu_reward_steps = run_pi2cma(start, sig2, sample_size, goal, scaler, reward_threshold, human_reward_threshold)

        print("----------")

        net = Net()
        net.load_state_dict(torch.load("data/model.pt"))
        net.eval()
        net_and_pi2cma_results, net_and_pi2cma_steps = run_net(start, sig2_hybrid, net, sample_size, goal, scaler, reward_threshold, human_reward_threshold)

        print("----------")

        print(f"Number of steps:\n\tPure PI2CMA: {pure_pi2cma_result}\n\tPI2CMA with Net: {net_and_pi2cma_results}")
        pure.append(pure_pi2cma_result)
        hybrid.append(net_and_pi2cma_results)

        net_steps, pi_steps = net_and_pi2cma_steps

        # net_steps = np.array([s[0] for s in net_steps])
        # pi_steps = np.array([s[0] for s in pi_steps])
        # pure_steps = np.array([s[0] for s in pure_mu_reward_steps])
        # print(net_steps)
        # print(pi_steps)
        # print(pure_steps)

    pure_filtered = []
    hybrid_filtered = []
    gained_convergences = 0
    lost_convergences = 0
    for r in range(len(pure)):
        if pure[r] is not None and hybrid[r] is not None:
            pure_filtered.append(pure[r])
            hybrid_filtered.append(hybrid[r])
        if pure[r] is None and hybrid[r] is not None:
            gained_convergences += 1
        if pure[r] is not None and hybrid[r] is None:
            lost_convergences += 1

    hybrid_filtered = np.array(hybrid_filtered)
    hybrid_avg = np.average(np.sum(hybrid_filtered, axis=1))
    pure_avg = np.average(pure_filtered)
    net_in_hybrid_avg = np.average(hybrid_filtered[:, 0])
    pi2cma_in_hybrid_avg = np.average(hybrid_filtered[:, 1])

    print(f"--------------------\nAverage number of steps:\n\tPure PI2CMA: {pure_avg}\n\tPI2CMA with "
          f"Net: {hybrid_avg}\n\t\tAvg Net Steps: {net_in_hybrid_avg}\n\t\tAvg PI2CMA Steps: {pi2cma_in_hybrid_avg}")
    print(f"\tOne Net instruction accounts for {(pure_avg-pi2cma_in_hybrid_avg)/net_in_hybrid_avg} samples")
    print(f"Percent Gained Convergences: {float(gained_convergences)/total_examples*100}%")
    print(f"Percent Lost Convergences: {float(lost_convergences) / total_examples * 100}%")


if __name__ == '__main__':
    main()
    # test()