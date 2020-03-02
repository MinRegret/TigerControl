"""Runs a random policy for the random object KukaObjectEnv.
"""

import tigercontrol
import numpy as np
import time
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
        return [dx, dy, dz, da, 0.5]


def test_kuka(verbose=False):

    environment = tigercontrol.environment("PyBullet-Kuka-v0")
    obs = environment.initialize(render=verbose)
    policy = ContinuousDownwardBiasPolicy()

    t_start = time.time()
    while time.time() - t_start < 5:
        done =  False
        episode_rew = 0
        while not done:
            if verbose:
                environment.render(mode='human')
            act = policy.sample_action(obs, .1)
            obs, rew, done, _ = environment.step(act)
            episode_rew += rew

    environment.close()
    print("test_kuka passed")

if __name__ == '__main__':
    #test_kuka(verbose=True)
    pass

