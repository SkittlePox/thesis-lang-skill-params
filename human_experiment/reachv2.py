import os
import numpy as np

from gym import utils
from gym.envs.robotics import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'slide.xml')


class FetchSlideEnvV2(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        initial_qpos = {
            'robot0:slide0': -0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.1,
            'object0:joint': [1.7, 1.1, 0.41, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.65, 0.0, 0.0]),
            obj_range=0.15, target_range=0.2, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2] + np.array([0.125, 0.])
            # while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            #     object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
            #                                                                          size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            # goal[0] = self.initial_gripper_xpos[0]
            goal = np.array([1.7, 0.77455443, 0.41401894])
            # goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        return goal.copy()