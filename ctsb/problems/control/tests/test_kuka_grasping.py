"""
Runs a random policy for the random object KukaDiverseObjectEnv.
"""

import numpy as np
import gym
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces


class ContinuousDownwardBiasPolicy(object):
    """Policy which takes continuous actions, and is biased to move down.
    """

    def __init__(self, height_hack_prob=0.9):
        """Initializes the DownwardBiasPolicy.

        Args:
                height_hack_prob: The probability of moving down at every move.
        """
        self._height_hack_prob = height_hack_prob
        self._action_space = spaces.Box(low=-1, high=1, shape=(5,))

    def sample_action(self, obs, explore_prob):
        """Implements height hack and grasping threshold hack.
        """
        dx, dy, dz, da, close = self._action_space.sample()
        if np.random.random() < self._height_hack_prob:
            dz = -1
        return [dx, dy, dz, da, 0]


def main():

    env = KukaDiverseObjectEnv(renders=True, isDiscrete=False)
    policy = ContinuousDownwardBiasPolicy()

    while True:
        done = False
        episode_rew = 0
        while not done:
            problem.render(mode='human')
            act = policy.sample_action(obs, .1)
            obs, rew, done, _ = problem.step([0, 0, 0, 0, 0])
            episode_rew += rew


if __name__ == '__main__':
    main()