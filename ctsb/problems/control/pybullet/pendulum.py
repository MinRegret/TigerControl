"""
PyBullet Pendulum enviornment
"""
import gym
import pybullet_envs
from ctsb.problems.control.pybullet.pybullet_problem import PyBulletProblem


class InvertedPendulumSwingupBulletEnv(PyBulletProblem):
    """
    Simulates a pendulum balanced on a cartpole.
    """
    def __init__(self):
        self.initialized = False

    def initialize(self):
        self.initialized = True
        problem = gym.make("InvertedPendulumSwingupBulletEnv-v0")
        problem.render(mode="human")
        self.problem = problem
        self.observation_space = problem.observation_space
        self.action_space = problem.action_space
        initial_obs = problem.reset()
        return initial_obs

    def step(self, a):
        return self.problem.step(a)

    def render(self, mode='human', close=False):
        self.problem.render(mode=mode, close=close)

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space



