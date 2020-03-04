"""
PyBullet Pendulum enviornment
"""

from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from tigercontrol.environments.deprecated.pybullet.pybullet_environment import PyBulletEnvironment


class Kuka(PyBulletEnvironment):
    """
    Description: Simulates a kuka arm picking up diverse objects
    """
    def __init__(self):
        self.initialized = False

    def initialize(self, render=False):
        self.initialized = True
        self._env = KukaGymEnv(renders=render)
        self.observation_space = self._env.observation_space.shape
        self.action_space = self._env.action_space.shape
        self.state = {}
        initial_obs = self.reset()
        return initial_obs





