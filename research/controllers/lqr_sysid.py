import matplotlib.pyplot as plt
import jax
import jax.numpy as np
import numpy as onp
import jax.random as random
import tigercontrol
from tigercontrol.utils.random import set_key, generate_key
from tigercontrol.environments import Environment
from tigercontrol.controllers import Controller
import scipy
from jax import grad,jit

class LQR_SystemID(Controller):
    """
    Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, n, m, x, T_0, K=None, Q = None, R = None):
        """
        Description: Initialize the dynamics of the model
        Args:
            n (float/numpy.ndarray): dimension of the state
            m (float/numpy.ndarray): dimension of the controls
            x (postive int): initial state
            T_0 (int): system identification time
            K  (float/numpy.ndarray): initial controller (optional)
            Q, R (float/numpy.ndarray): cost matrices (c(x, u) = x^T Q x + u^T R u)
        """
        self.initialized = True

        self.n, self.m = n, m
        self.x = x
        self.T_0 = T_0
        self.T = 0

        Q = np.identity(n) if Q is None else Q
        R = np.identity(m) if R is None else R

        if K:
            self.K = K
        else:
            X = scipy.linalg.solve_discrete_are(A, B, Q, R)
            self.K = np.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    def get_action(self):
        #choose action
        if self.T < self.T_0:
            return
        self.u = -self.K @ self.x
            
        return self.u

    def update(self, c_t, x_new):
        """
        Description: Updates internal parameters and then returns the estimated optimal action (only one)
        Args:
            None
        Returns:
            Estimated optimal action
        """
        self.T += 1
        self.x = x_new

        return