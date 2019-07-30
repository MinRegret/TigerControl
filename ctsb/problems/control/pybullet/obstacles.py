"""
PyBullet obstacles enviornment
"""

import pybullet as pybullet
from ctsb.problems.control.pybullet.pybullet_problem import PyBulletProblem
from ctsb.problems.control.pybullet.obstacle_utils import *
from ctsb.problems.control.pybullet.obstacles_env import ObstaclesEnv


class Obstacles(PyBulletProblem):
    """
    Description: Simulates a obstacles avoidance environment
    """

    compatibles = set(['Obstacles-v0', 'PyBullet'])

    def __init__(self):
        self.initialized = False

    def initialize(self, render=False):
        self.initialized = True
        self._env = ObstaclesEnv(renders=render)
        self.observation_space = self._env.observation_space.shape
        self.action_space = self._env.action_space.shape
        self.state = None
        state, params, husky, sphere, obsUid = self._env.reset()
        return (state, params, husky, sphere, obsUid)

    def step(self, action):
        return self._env.step(action)

    def step_fgm(self, angle):
        return self._env.step_fgm(angle)

    def render(self, mode='human', close=False):
        self._env.render(mode=mode, close=close)





