"""
PyBullet Pendulum enviornment
"""

from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from tigercontrol.environments.deprecated.pybullet.pybullet_environment import PyBulletEnvironment


class KukaDiverse(PyBulletEnvironment):
    """
    Description: Simulates a kuka arm picking up diverse objects
    """
    def __init__(self):
        self.initialized = False

    def initialize(self, render=False):
        self.initialized = True
        self._env = KukaDiverseObjectEnv(renders=render)
        self.observation_space = (48,48) # observation image dimensions
        self.action_space = self._env.action_space.shape
        self.state = {}
        initial_obs = self.reset()
        return initial_obs


