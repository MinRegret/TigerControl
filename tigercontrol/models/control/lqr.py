"""
Linear Quadratic Regulator
"""

import jax.numpy as np
import tigercontrol
from tigercontrol.models.control import ControlModel

class LQR(ControlModel):
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
        x = self.to_ndarray(x)
        return [x for i in range(T)]

    def initialize(self, F, f, C, c, T, x):
        """
        Description: Initialize the dynamics of the model
        Args:
            F (float/numpy.ndarray): past value contribution coefficients
            f (float/numpy.ndarray): bias coefficients
            C (float/numpy.ndarray): quadratic cost coefficients
            c (float/numpy.ndarray): linear cost coefficients
            T (postive int): number of timesteps
            x (float/numpy.ndarray): initial state
        """
        self.initialized = True
        
        self.F, self.f, self.C, self.c, self.T, self.x = self.extend(F, T), self.extend(f, T), self.extend(C, T), self.extend(c, T), T, self.to_ndarray(x)
        
        self.u = self.extend(np.zeros((self.F[0].shape[1] - self.x.shape[0], 1)), T)
        
        self.K = self.extend(np.zeros((self.u[0].shape[0], self.x.shape[0])), T)
        self.k = self.u.copy()

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
        v = np.zeros((self.F[0].shape[0], 1))
        Q = np.zeros((self.C[0].shape[0], self.C[0].shape[1]))
        q = np.zeros((self.c[0].shape[0], 1))

        ## Backward Recursion ##
        for t in range(self.T - 1, -1, -1):

            Q = self.C[t] + self.F[t].T @ V @ self.F[t]
            q = self.c[t] + self.F[t].T @ V @ self.f[t] + self.F[t].T @ v

            self.K[t] = -np.linalg.inv(Q[self.x.shape[0] :, self.x.shape[0] :]) @ Q[self.x.shape[0] :, : self.x.shape[0]]
            self.k[t] = -np.linalg.inv(Q[self.x.shape[0] :, self.x.shape[0] :]) @ q[self.x.shape[0] :]

            V = Q[: self.x.shape[0], : self.x.shape[0]] + Q[: self.x.shape[0], self.x.shape[0] :] @ self.K[t] + self.K[t].T @ Q[self.x.shape[0] :, : self.x.shape[0]] + self.K[t].T @ Q[self.x.shape[0] :, self.x.shape[0] :] @ self.K[t]
            v = q[: self.x.shape[0]] + Q[: self.x.shape[0], self.x.shape[0] :] @ self.k[t] + self.K[t].T @ q[self.x.shape[0] :] + self.K[t].T @ Q[self.x.shape[0] :, self.x.shape[0] :] @ self.k[t]

        ## Forward Recursion ##
        for t in range(self.T):
            self.u[t] = self.K[t] @ self.x + self.k[t]
            self.x = self.F[t] @ np.vstack((self.x, self.u[t])) + self.f[t]

        return self.u


    def help(self):
        """
        Description: Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(LQR_help)

    def __str__(self):
        return "<LQR Model>"


# string to print when calling help() method
LQR_help = """

-------------------- *** --------------------

Id: LQR

Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.

Methods:

    initialize(F, f, C, c, T, x)
        Description:
            Initialize the dynamics of the model
        Args:
            F (float/numpy.ndarray): past value contribution coefficients
            f (float/numpy.ndarray): bias coefficients
            C (float/numpy.ndarray): quadratic cost coefficients
            c (float/numpy.ndarray): linear cost coefficients
            T (postive int): number of timesteps
            x (float/numpy.ndarray): initial state

    step()
        Description: Updates internal parameters and then returns the
        	estimated optimal set of actions
        Args:
            None
        Returns:
            Estimated optimal set of actions

    predict()
        Description:
            Returns estimated optimal set of actions
        Args:
            None
        Returns:
            Estimated optimal set of actions

    update()
        Description:
        	Updates internal parameters
        Args:
            None

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""