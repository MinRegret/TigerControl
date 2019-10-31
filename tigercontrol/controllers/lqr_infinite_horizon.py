import jax.numpy as np
import tigercontrol
from tigercontrol.controllers import Controller
from jax import grad,jit
import jax.random as random
from tigercontrol.utils import generate_key
import jax
import scipy

class LQRInfiniteHorizon(Controller):

    def __init__(self):
        self.initialized = False

    def _get_dims(self):
        try:
            n = self.B.shape[0] ## dimension of  the state x 
        except:
            n = 1
        try:
            m = self.B.shape[1] ## dimension of the control u
        except:
            m = 1
        return (n, m)

    def initialize(self, A, B, Q = None, R = None):
        """
        Description: Initialize the infinite-time horizon LQR.
        Args:
            A, B (float/numpy.ndarray): system dynamics
            Q, R (float/numpy.ndarray): cost matrices (i.e. cost = x^TQx + u^TRu)
        """
        self.initialized = True

        self.A, self.B = A, B
        n, m = self._get_dims()

        if(Q is None):
            Q = np.identity(n)
        if(R is None):
            R = np.identity(m)

        # solve the ricatti equation 
        X = scipy.linalg.solve_continuous_are(A, B, Q, R)

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
        return self.K @ x

    def update(self):
        raise NotImplementedError

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
