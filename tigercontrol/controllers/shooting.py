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

    def initialize(self, env, optimizer):
        """
        Description: Initialize the dynamics of the controller.
        Args:
            f (function): describes dy/dt = f(y,t)
            a (float): value of y(0)
            b (float): value of y(L)
            z1 (float): first initial estimate of y'(0)
            z2 (float): second initial estimate of y'(0)
            t (float): time value to determine y at
        """
        self.env = env

        self.initialized = True

    def plan(self, x, T):
        pass


    def __str__(self):
        return "<Shooting Controller>"
