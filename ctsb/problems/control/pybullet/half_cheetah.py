"""
PyBullet HalfCheetah enviornment
"""

from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv
from ctsb.problems.control.pybullet.pybullet_problem import PyBulletProblem


class HalfCheetah(PyBulletProblem):
    """
    Simulates a minitaur environment
    """
    def initialize(self, render=False):
        self.initialized = True
        self._env = HalfCheetahBulletEnv(render=render)
        self.observation_space = self._env.observation_space.shape
        self.action_space = self._env.action_space.shape
        self.state = {}
        initial_obs = self.reset()
        return initial_obs
