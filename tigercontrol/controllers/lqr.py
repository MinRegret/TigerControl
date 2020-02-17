import jax.numpy as np
import numpy as onp
import tigercontrol
from tigercontrol.controllers import Controller
from jax import grad,jit
import jax.random as random
from tigercontrol.utils import generate_key
import jax
from scipy.linalg import solve_discrete_are as dare

class LQR(Controller):

    def __init__(self, A, B, Q = None, R = None):
        """
        Description: Initialize the infinite-time horizon LQR.
        Args:
            A, B (float/numpy.ndarray): system dynamics
            Q, R (float/numpy.ndarray): cost matrices (i.e. cost = x^TQx + u^TRu)
        """

        n, m = B.shape # State & Action Dimensions

        if(Q is None or type(Q)):
            Q = onp.identity(n, dtype=np.float32)
        if(R is None):
            R = onp.identity(m, dtype=np.float32)

        # solve the ricatti equation 
        X = dare(A, B, Q, R)

        #compute LQR gain
        self.K = np.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    def get_action(self, x):
        """
        Description: Return the action based on current state and internal parameters.

        Args:
            x (float/numpy.ndarray): current state

        Returns:
            u(float/numpy.ndarray): action to take
        """
        return -self.K @ x

    def update(self, cost = None):
        return

    def plan(self, x, T):
        """
        Description: Plan next T actions.

        Args:
            x (float/numpy.ndarray): starting state
            T (int): number of timesteps to plan actions for
            
        Returns:
            u (list): list of actions to take
        """
        u = []
        for i in range(T):
            u.append(self.get_action(x))
            x = self.A @ x + self.B @ u[i]
        return u
