"""
Online Least Squares
"""

import tigercontrol
import jax.numpy as np
from tigercontrol.methods.time_series import TimeSeriesMethod

class LeastSquares(TimeSeriesMethod):
    """
    Description: Implements online least squares.
    """
    
    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.uses_regressors = True

    def initialize(self, x, y, reg = 0.0):
        """
        Description: Initializes method parameters

        Args:
            x (float/numpy.ndarray): Observation
            y (float): label
            reg (float): regularization factor
        """
        self.initialized = True
        self.Pinv = np.linalg.inv(reg * np.eye(x.shape[0]) + np.outer(x, x))
        self.w = self.Pinv @ x * y

    def step(self, x, y):
        """
        Description: Predict next value given observation and update internal parameters based
                     on true label.

        Args:
            x (float/numpy.ndarray): Observation
            y (float): label
        Returns:
            Predicted value for the next time-step
        """
        assert self.initialized, "ERROR: Method not initialized!"
        
        # get prediction
        pred = np.dot(self.w, x)

        # update 
        self.w += self.Pinv @ x * (y - pred) / (1 + x.T @ self.Pinv @ x)
        self.Pinv -= (self.Pinv @ x) @ (self.Pinv @ x).T / (1 + x.T @ self.Pinv @ x)

        return pred
