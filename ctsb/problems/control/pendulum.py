#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
# import os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0, parentdir)

import gym
# import numpy as np
# import pybullet_envs
# import time

from ctsb.problems.control.control_problem import ControlProblem

class InvertedPendulumSwingupBulletEnv(ControlProblem):
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



