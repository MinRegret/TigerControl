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

class LQR(Controller):
    """
    Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, A, B, n, m, x, Q, R):
        """
        Description: Initialize the dynamics of the model
        Args:
            A,B (float/numpy.ndarray): system dynamics
            K  (float/numpy.ndarray): optimal controller 
            n (float/numpy.ndarray): dimension of the state
            m (float/numpy.ndarray): dimension of the controls
            H (postive int): history of the controller 
            Q, R (float/numpy.ndarray): cost matrices (c(x, u) = x^T Q x + u^T R u)
        """
        self.initialized = True

        self.x = x        

        X = scipy.linalg.solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    def get_action(self):
        
        #choose action
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

        self.x = x_new

        return
