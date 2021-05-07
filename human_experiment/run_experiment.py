import gym
import skills_kin
import torch
from skills_kin.utils import dmps_from_scratch, fit_dmps, run_gym_episode
from skills_kin.agent_dmp import Agent_DMP
from skills_kin.pi2cma import run_pi2cma, assign_weights
import numpy as np
from reachv2 import FetchSlideEnvV2
import pickle

rng = np.random.default_rng()

# env = FetchSlideEnvV2(goal=np.array([1.489, 0.57, 0.41401894]))
DEFAULT_DIR = 'test1'


def run_experiment(env, agent, vis=False):
    agent.reset()
    obs = env.reset()
    print(obs)
    if vis:
        env.render()

    for i in range(70):
        if vis:
            env.render()

        action = agent.act(i)
        obs = env.step(action)
        print(obs)


# def run_experiment():
#     obs = env.reset()
#     print(obs)
#     env.render()
#
#     for i in range(1000):
#         env.render()
#         # action = env.action_space.sample()
#         # print(action['desired_goal'])
#         # print(action)
#         # if i%50==0:
#         #   env.reset()
#         obs = env.step([0, 0, 0, 0])  # take a random action
#         # print(len(outt))
#         # print(type(outt))
#         # print(obs)


# def run_experiment_with_agent(agent):
#     obs = env.reset()
#     agent.reset()
#
#     for i in range(70):
#         env.render()
#         obs = env.step(agent.act(i))
#         print(obs)


def train_dmp_agent(env):
    demo_data = [[3.0, 0.0, 0.0, 0.0]] * 69
    demo_data.append([3.1, 0.1, 0.1, 0.1])
    demo_data = np.array(demo_data)
    demo_data = demo_data.transpose()
    dmps = fit_dmps(demo_data, demo_data.shape[0], 10, 10, 4, 1.0)
    # f = dmps[0].step()
    # print(f)

    for d in dmps:
        d.cs.N = 70

    agent = Agent_DMP(dmps, 10, mode='position')

    nw = 10 * 4
    tau_var = 10.0

    goal_var = np.pi / 2 * 100

    sigma = np.eye(nw + 4 + 1)
    sigma[:nw, :nw] = sigma[:nw, :nw] * 100
    sigma[nw:nw + 4, nw:nw + 4] = sigma[nw:nw + 4, nw:nw + 4] * goal_var
    sigma[-1, -1] = tau_var

    agent, episode_rewards, eval_rewards = run_pi2cma(env,
                                                      agent,
                                                      run_gym_episode,
                                                      30,
                                                      15,
                                                      70,
                                                      20,
                                                      sigma,
                                                      True,
                                                      True,
                                                      name='test1',
                                                      vis=False)
    agent.reset()
    # run_experiment_with_agent(agent)


def generate_random_goal():
    goal = np.array([0.8509234, 0.7491009, 0.45600106]) + rng.uniform(-0.2, 0.2, size=3) + np.array([0.65, 0.0, 0.0])
    goal[2] = 0.41401894
    return goal


def run_random_experiments(count=1):
    # goal = np.array([1.489, 0.57, 0.41401894])  # This is the object goal position
    agent = get_pickled_agent()
    for i in range(count):
        env = FetchSlideEnvV2(goal=np.array([1.489, 0.57, 0.41401894]))
        run_experiment(env, agent)


def get_pickled_agent(fn=None):
    if fn is None:
        fn = DEFAULT_DIR
    fl = open(f'{fn}/agent.obj', 'rb')
    return pickle.load(fl)


def extract_agent_params(agent):
    weights = []
    for i in range(len(agent.dmps)):
        weights.append(agent.dmps[i].w)
        # print(f'weights[{i}]: {agent.dmps[i].w}')

    params = np.array([weights]).flatten()
    params = np.concatenate((params, np.array(agent.goals), np.array([agent.tau])))
    return params


def main():
    # run_random_experiments()
    agent = get_pickled_agent()
    # env = FetchSlideEnvV2(goal=np.array([1.489, 0.57, 0.41401894]))
    # run_experiment(env, agent, True)
    # fl = open('test1/agent.obj', 'rb')
    # agent = pickle.load(fl)
    # run_experiment_with_agent(agent)
    # get_dmp_agent()
    # run_experiment()
    # env.reset()
    # env.reset()
    # print(generate_random_goal())


        # agent.goals = [d.g for d in agent.dmps]

    # if sample_tau:
    #     agent.tau = w_episode[-1]

    params = extract_agent_params(agent)
    print(params)
    # agent2 = Agent _DMP()
    assign_weights(agent, params, True, True)
    # print(params[0])
    # print(params[1])
    # print(params[2])



if __name__ == "__main__":
    main()
