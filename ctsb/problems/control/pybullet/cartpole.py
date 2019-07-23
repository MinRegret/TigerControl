"""
PyBullet Pendulum enviornment
"""
import gym
import pybullet_envs
from ctsb.problems.control.pybullet.pybullet_problem import PyBulletProblem
from ctsb.problems.control.pybullet.simulator_wrapper import SimulatorWrapper


class CartPole(PyBulletProblem):
    """
    Description: Simulates a pendulum balanced on a cartpole.
    """

    compatibles = set(['CartPole-v0', 'PyBullet'])
    
    def __init__(self):
        self.initialized = False

    def initialize(self, render=False):
        self.initialized = True
        self._env = gym.make("InvertedPendulumBulletEnv-v0")
        if render:
            self._env.render(mode="human")
        self.sim = SimulatorWrapper(self._env)
        self.observation_space = self._env.observation_space.shape
        self.action_space = self._env.action_space.shape
        initial_obs = self._env.reset()
        return initial_obs

    def step(self, action):
        return self.sim.step(action)

    def render(self, mode='human', close=False):
        self._env.render(mode=mode, close=close)

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space



