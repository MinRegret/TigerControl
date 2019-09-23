"""
PyBullet Swingup Pendulum enviornment
"""

from pybullet_envs.gym_pendulum_envs import InvertedPendulumSwingupBulletEnv
from tigercontrol.problems.pybullet.pybullet_problem import PyBulletProblem


class CartPoleSwingup(PyBulletProblem):
    """
    Description: Simulates a pendulum balanced on a cartpole.
    """

    compatibles = set(['CartPoleSwingup-v0', 'PyBullet'])
    
    def __init__(self):
        self.initialized = False

    def initialize(self, render=False):
        self.initialized = True
        self._env = InvertedPendulumSwingupBulletEnv()
        if render:
            self._env.render(mode="human")
        self.observation_space = self._env.observation_space.shape
        self.action_space = self._env.action_space.shape
        initial_obs = self._env.reset()
        return initial_obs

    def step(self, a):
        return self._env.step(a)

    def render(self, mode='human', close=False):
        self._env.render(mode=mode, close=close)
