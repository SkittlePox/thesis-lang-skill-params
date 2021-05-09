import os
import gym
import numpy
import skills_kin
import torch
from skills_kin.utils import dmps_from_scratch, fit_dmps, run_gym_episode
from skills_kin.agent_dmp import Agent_DMP
from skills_kin.pi2cma import run_pi2cma, assign_weights
import numpy as np
from reachv2 import FetchSlideEnvV2
import pickle

rng = np.random.default_rng()

DEFAULT_DIR = 'agents/default_dir'


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


def get_achieved_goal(env, agent):
    agent.reset()
    env.reset()

    for i in range(70):
        action = agent.act(i)
        obs = env.step(action)
    return obs[0]['achieved_goal']


def view_agent_runs(name, count):
    for c in range(count):
        agent, env = get_pickled_agent_env(f'{name}_{c}')

        run_experiment(env, agent, vis=True)


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


def train_dmp_agent(env, name=DEFAULT_DIR):
    try:
        os.mkdir(name)
    except FileExistsError:
        pass

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
                                                      15,
                                                      15,
                                                      70,
                                                      20,
                                                      sigma,
                                                      True,
                                                      True,
                                                      name=name,
                                                      vis=False)
    # print(episode_rewards)
    # print(eval_rewards)
    agent.reset()
    # filehandler_env = open('%s/env.obj' % name, 'wb')
    # pickle.dump(env, filehandler_env)
    return agent
    # run_experiment_with_agent(agent)


def generate_random_goal():
    goal = np.array([0.8509234, 0.7491009, 0.45600106]) + rng.uniform(-0.2, 0.2, size=3) + np.array([0.65, 0.0, 0.0])
    goal[2] = 0.41401894
    return goal


def run_random_experiments(count=10):
    # goal = np.array([1.489, 0.57, 0.41401894])  # This is the object goal position
    agent = get_pickled_agent_env()[0]
    for i in range(count):
        env = FetchSlideEnvV2(goal=generate_random_goal())
        run_experiment(env, agent, True)


def get_pickled_agent_env(fn=None):
    if fn is None:
        fn = DEFAULT_DIR
    fa = open(f'{fn}/agent.obj', 'rb')
    fe = open(f'{fn}/env.obj', 'rb')
    return pickle.load(fa), pickle.load(fe)


def extract_agent_params(agent):
    weights = []
    for i in range(len(agent.dmps)):
        weights.append(agent.dmps[i].w)
        # print(f'weights[{i}]: {agent.dmps[i].w}')

    params = np.array([weights]).flatten()
    params = np.concatenate((params, np.array(agent.goals), np.array([agent.tau])))
    return params


def collect_data(datapoints=1, name=DEFAULT_DIR):
    goals = []
    params = []
    for d in range(datapoints):
        # collect a data point [achieved goal, policy parameters]

        # create a random new goal
        goal = generate_random_goal()

        # create environment from goal
        env = FetchSlideEnvV2(goal=goal)

        # train new dmp agent
        agent = train_dmp_agent(env, name=f'{name}_{d}')

        goals.append(get_achieved_goal(env, agent))
        params.append(extract_agent_params(agent))

    return np.array(goals), np.array(params)


def main():
    pass
    # fl = open('data/goals_0', 'rb')
    # goals = pickle.load(fl)
    # # goals = numpy.array(goals)
    #
    # fel = open('data/params_0', 'rb')
    # params = pickle.load(fel)
    # # params = numpy.array(params)
    #
    # print(goals)
    # print(params)

    # goals, params = collect_data(1000, name='agents/exp2_agent')
    #
    # filehandler_g = open('data/goals_1', 'wb')
    # pickle.dump(goals, filehandler_g)
    #
    # filehandler_p = open('data/params_1', 'wb')
    # pickle.dump(params, filehandler_p)

    # view_agent_runs('agents/exp2_agent', 10)

    run_random_experiments()

    # goal = generate_random_goal()
    # #
    # # # create environment from goal
    # env = FetchSlideEnvV2(goal=goal)

    # train new dmp agent
    # agent = train_dmp_agent(env, name='exp1_agent_0')
    # run_experiment(env, agent, True)

    # agent, env2 = get_pickled_agent_env('exp1_agent_0')
    #
    # p = get_achieved_goal(env2, agent)
    # print(p)


    # run_experiment(env, agent, True)
    # env.reset()
    # run_experiment(env, agent, True)

    # params = extract_agent_params(agent)
    #
    # assign_weights(agent, params, True, True)
    # run_experiment(env2, agent, True)




    # goals, params = collect_data()
    #
    # env = FetchSlideEnvV2(goal=goals[0])
    #

    # assign_weights(agent, params[0], True, True)
    #
    # run_experiment(env, agent, True)



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

    # params = extract_agent_params(agent)
    # print(params)
    # # agent2 = Agent _DMP()
    # assign_weights(agent, params, True, True)
    # print(params[0])
    # print(params[1])
    # print(params[2])



if __name__ == "__main__":
    main()
