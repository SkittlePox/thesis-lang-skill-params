import gym
import skills_kin
import torch
from skills_kin.utils import dmps_from_scratch, fit_dmps, run_gym_episode
from skills_kin.agent_dmp import Agent_DMP
from skills_kin.pi2cma import run_pi2cma
import numpy as np
from reachv2 import FetchSlideEnvV2

env = FetchSlideEnvV2()


def run_experiment():
    obs = env.reset()
    print(obs)

    for i in range(1000):
        env.render()
        # action = env.action_space.sample()
        # print(action['desired_goal'])
        # print(action)
        # if i%50==0:
        #   env.reset()
        obs = env.step([3, 0, 0, 0])  # take a random action
        # print(len(outt))
        # print(type(outt))
        print(obs)


def run_slide_episode(env,
                      agent,
                      agent_type='dmp',
                      n_steps=int(1e3),
                      vis=False,
                      feedback=False):
    rewards = []
    obs = env.reset()
    agent.reset()

    input_dim = 4

    # %% start state
    # start = env.home
    # for i in range(len(start)):
    #     p.resetJointState(env.kuka.kukaUid, i, start[i])

    for t in range(n_steps):
        if feedback:
            obs_float = obs.astype(float)
            obs_torch = torch.from_numpy(obs_float)
            obs_input = obs_torch.view(1, input_dim).float()
            action = agent.act(obs_input)
        else:
            action = agent.act(t)
        # print(action)
        obs, reward, done, info = env.step(action)

        rewards.append(reward)
    return rewards


def get_dmp_agent():
    # demo_data = np.zeros((4, 100))
    # print(demo_data.shape)
    demo_data = [[3.0, 0.0, 0.0, 0.0]] * 99
    demo_data.append([3.1, 0.1, 0.1, 0.1])
    demo_data = np.array(demo_data)
    demo_data = demo_data.transpose()
    # print(demo_data.shape)
    # print(demo_data[-1])
    dmps = fit_dmps(demo_data, demo_data.shape[0], 10, 10, 4, 1.0)
    # f = dmps[0].step()
    # print(f)

    for d in dmps:
        d.cs.N = 100
    # goal = np.array([1.65247544, 0.63924951, 0.41401894])

    agent = Agent_DMP(dmps, 10, mode='position')

    nw = 10 * 4
    tau_var = 0.01

    goal_var = np.pi / 2

    sigma = np.eye(nw + 4 + 1)
    sigma[:nw, :nw] = sigma[:nw, :nw] * 1.0
    sigma[nw:nw + 4, nw:nw + 4] = sigma[nw:nw + 4, nw:nw + 4] * goal_var
    sigma[-1, -1] = tau_var

    print(sigma.shape)

    agent, episode_rewards, eval_rewards = run_pi2cma(env,
                                                      agent,
                                                      run_gym_episode,
                                                      10,
                                                      16,
                                                      60,
                                                      40,
                                                      sigma,
                                                      True,
                                                      True,
                                                      name='test1',
                                                      vis=True)


def main():
    # get_dmp_agent()
    run_experiment()
    # env.reset()


if __name__ == "__main__":
    main()
