"""
Linear Quadratic Regulator
"""

import jax.numpy as np
import tigercontrol
from tigercontrol.methods import Method

class LQR(Method):
    """
    Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.
    """
    
    compatibles = set([])

    def __init__(self):
        self.initialized = False

    def to_ndarray(self, x):
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

    def extend(self, x, T):
        """
        Description: If x is not in the correct form, convert it; otherwise, leave it unchanged.
        Args:
            x (float/numpy.ndarray)
            T (postive int): number of timesteps
        Returns:
            A numpy.ndarray representation of x
        """
        #x = self.to_ndarray(x)
        return [x for i in range(T)]

    def initialize(self, A, B, C, T, x):
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

        n, m = B.shape[0], B.shape[1]
        
        self.F, self.C, self.T, self.x = self.extend(F, T), self.extend(C, T), T, self.to_ndarray(x)
        self.u = self.extend(np.zeros((m, 1)), T)     
        self.K = self.extend(np.zeros((m, n)), T)

        self.is_online = False

    def plan(self):
        """
        Description: Updates internal parameters and then returns the estimated optimal set of actions
        Args:
            None
        Returns:
            Estimated optimal set of actions
        """

        ## Initialize V and Q Functions ##
        V = np.zeros((self.F[0].shape[0], self.F[0].shape[0]))
        Q = np.zeros((self.C[0].shape[0], self.C[0].shape[1]))

        ## Backward Recursion ##
        for t in range(self.T - 1, -1, -1):

            Q = self.C[t] + self.F[t].T @ V @ self.F[t]
            self.K[t] = -np.linalg.inv(Q[self.x.shape[0] :, self.x.shape[0] :]) @ Q[self.x.shape[0] :, : self.x.shape[0]]
            V = Q[: self.x.shape[0], : self.x.shape[0]] + Q[: self.x.shape[0], self.x.shape[0] :] @ self.K[t] + self.K[t].T @ Q[self.x.shape[0] :, : self.x.shape[0]] + self.K[t].T @ Q[self.x.shape[0] :, self.x.shape[0] :] @ self.K[t]
        
        ## Forward Recursion ##
        for t in range(self.T):
            self.u[t] = self.K[t] @ self.x
            self.x = self.F[t] @ np.hstack((self.x, self.u[t]))

        return self.u

    def __str__(self):
        return "<LQR Method>"


