"""
PyBullet Humanoid enviornment
"""

from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv
from tigercontrol.environments.pybullet.pybullet_environment import PyBulletEnvironment


class Humanoid(PyBulletEnvironment):
    """
    Description: Simulates a minitaur environment
    """
    def initialize(self, render=False):
        self.initialized = True
        self._env = HumanoidBulletEnv(render=render)
        self.observation_space = self._env.observation_space.shape
        self.action_space = self._env.action_space.shape
        self.state = {}
        initial_obs = self.reset()
        return initial_obs
