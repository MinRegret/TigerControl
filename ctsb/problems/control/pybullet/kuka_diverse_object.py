"""
PyBullet Pendulum enviornment
"""
import os, inspect
import numpy as np
import gym
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces
from ctsb.problems.control.control_problem import ControlProblem


class Kuka_diverse_object(ControlProblem):
    """
    Simulates a pendulum balanced on a cartpole.
    """
    def __init__(self):
        self.initialized = False

    def initialize(self):
        self.initialized = True

        problem = KukaDiverseObjectEnv(renders=True, isDiscrete=False)

        # problem = gym.make("InvertedPendulumSwingupBulletEnv-v0")
        # problem.render(mode="human")
        self.problem = problem
        self.observation_space = problem.observation_space
        self.action_space = problem.action_space
        initial_obs = problem.reset()
        return initial_obs

    def step(self, action):
        """Environment step.
        Args:
          action: 5-vector parameterizing XYZ offset, vertical angle offset
          (radians), and grasp angle (radians).
        Returns:
          observation: Next observation.
          reward: Float of the per-step reward as a result of taking the action.
          done: Bool of whether or not the episode has ended.
          debug: Dictionary of extra information provided by environment.
        """
        dv = self.problem._dv  # velocity per physics step.
        if self.problem._isDiscrete:
          # Static type assertion for integers.
          assert isinstance(action, int)
          if self.problem._removeHeightHack:
            dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0][action]
            dz = [0, 0, 0, 0, 0, -dv, dv, 0, 0][action]
            da = [0, 0, 0, 0, 0, 0, 0, -0.25, 0.25][action]
          else:
            dx = [0, -dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -dv, dv, 0, 0][action]
            dz = -dv
            da = [0, 0, 0, 0, 0, -0.25, 0.25][action]
        else:
          dx = dv * action[0]
          dy = dv * action[1]
          if self.problem._removeHeightHack:
            dz = dv * action[2]
            da = 0.25 * action[3]
          else:
            dz = dv
            da = 0.25 * action[2]

        return self.problem._step_continuous([dx, dy, dz, da, 0.3])
        # return self.problem.step2(a)

    def render(self, mode='human', close=False):
        self.problem.render(mode=mode, close=close)

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space



