import gym
import numpy as np

env = gym.make('FetchReach-v1')


def run_experiment():
    env.reset()
    for i in range(1000):
        env.render()
        action = env.action_space.sample()
        # print(action)
        outt = env.step(action)  # take a random action
        print(outt)


def main():
    run_experiment()


if __name__ == "__main__":
    main()
