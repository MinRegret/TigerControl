"""
Linear Quadratic Regulator
"""

import jax.numpy as np
import tigercontrol
from tigercontrol.controllers import Controller

class LQRFiniteHorizon(Controller):
    """
    Description: Computes optimal set of actions for finite horizon using the Linear Quadratic Regulator
    algorithm.
    """

    def __init__(self, A, B, C, T):
        """
        Description: Initialize the dynamics of the method

        Args:
            A (float/numpy.ndarray): past value contribution coefficients
            B (float/numpy.ndarray): control value contribution coefficients
            C (float/numpy.ndarray): quadratic cost coefficients
            T (postive int): number of timesteps
            x (float/numpy.ndarray): initial state
        """
        self.initialized = True
        
        F = np.hstack((A, B))

         n, m = B.shape # State & Action Dimensions
        
        self.F, C, self.T = self._extend(F, T), self._extend(C, T), T
        self.K = self._extend(np.zeros((self.m, n)), T)

        ## Initialize V and Q Functions ##
        V = np.zeros((self.F[0].shape[0], self.F[0].shape[0]))
        Q = np.zeros((C[0].shape[0], C[0].shape[1]))

        ## Backward Recursion ##
        for t in range(self.T - 1, -1, -1):

            Q = C[t] + self.F[t].T @ V @ self.F[t]
            self.K[t] = -np.linalg.inv(Q[n :, n :]) @ Q[n :, : n]
            V = Q[: n, : n] + Q[: n, n :] @ self.K[t] + self.K[t].T @ Q[n :, : n] + self.K[t].T @ Q[n :, n :] @ self.K[t]

    def _to_ndarray(self, x):
        """
        Description: If x is a scalar, transform it to a (1, 1) numpy.ndarray;
        otherwise, leave it unchanged.
        Args:
            x (float/numpy.ndarray)
        Returns:
            A numpy.ndarray representation of x
        """
        x = np.asarray(x)
        if(np.ndim(x) == 0):
            x = x[None, None]
        return x

    def _extend(self, x, T):
        """
        Description: If x is not in the correct form, convert it; otherwise, leave it unchanged.
        Args:
            x (float/numpy.ndarray)
            T (postive int): number of timesteps
        Returns:
            A numpy.ndarray representation of x
        """
        #x = self._to_ndarray(x)
        return [x for i in range(T)]

    def plan(self, x, T):
        """
        Description: Updates internal parameters and then returns the estimated optimal set of actions
        Args:
            None
        Returns:
            Estimated optimal set of actions
        """

        assert T == self.T, "ERROR: Can only plan for the initial timeline provided."

        u = self._extend(np.zeros((self.m, 1)), T) 

        ## Forward Recursion ##
        for t in range(T):
            u[t] = self.K[t] @ x
            x = self.F[t] @ np.hstack((x, u[t]))

        return u

    def __str__(self):
        return "<Finite Horizion LQR Method>"

