"""
PyBullet Pendulum enviornment
"""

from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from ctsb.problems.control.pybullet.pybullet_problem import PyBulletProblem


class KukaDiverse(PyBulletProblem):
    """
    Simulates a kuka arm picking up diverse objects
    """
    def __init__(self):
        self.initialized = False

    def initialize(self):
        self.initialized = True
        self._env = KukaDiverseObjectEnv()
        self.observation_space = self._env.observation_space.shape
        self.action_space = self._env.action_space.shape
        self.state = {}
        initial_obs = self.reset()
        return initial_obs


