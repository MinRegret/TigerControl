"""
PyBullet Pendulum enviornment
"""

from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv
from ctsb.problems.control.pybullet.pybullet_problem import PyBulletProblem


class Minitaur(PyBulletProblem):
    """
    Simulates a minitaur environment
    """
    def initialize(self, render=False):
        self.initialized = True
        self._env = MinitaurBulletEnv(render=render)
        self.observation_space = self._env.observation_space.shape
        self.action_space = self._env.action_space.shape
        self.state = {}
        initial_obs = self.reset()
        return initial_obs
