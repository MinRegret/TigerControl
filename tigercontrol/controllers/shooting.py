"""
ODE Shooting Controller
"""

import jax.numpy as np
import tigercontrol
from tigercontrol.controllers import Controller

class Shooting(Controller):
    """
    Description: Implements the shooting controller to solve second order boundary value
    environments with conditions y(0) = a and y(L) = b. Assumes that the
    second order BVP has been converted to a first order system of two
    equations.
    """

    compatibles = set([])

    def __init__(self):
        self.initialized = False
        class BabyController(Controller): # used to get trajectories from env.rollout
            def __init__(self):
                pass
            def get_action(self, x):
                return x

    def initialize(self, n, m, env, optimizer, update_steps=25):
        """
        Description: Initialize the dynamics of the controller.
        Args:
            n (int): observation dimension
            m (int): action dimension
            env (Environment): task on which to act on
            optimizer (Optimizer): optimizer that performs gradient descent
            update_steps (int): number of times to perform a gradient update step
        """
        self.n, self.m = n, m
        self.env = env
        self._store_optimizer(optimizer, pred) # what is PRED????
        self.initialized = True

    def plan(self, x, T):
        u = np.zeros((T, self.m)) # T copies of m-dimensional zero action vector
        # trajectory = 


    def __str__(self):
        return "<Shooting Controller>"


