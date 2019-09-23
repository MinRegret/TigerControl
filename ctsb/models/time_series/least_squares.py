"""
Online Least Squares
"""

import tigercontrol
import jax.numpy as np
from tigercontrol.models.time_series import TimeSeriesModel

class LeastSquares(TimeSeriesModel):
    """
    Description: Implements online least squares.
    """
    
    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.uses_regressors = True

    def initialize(self, x, y, reg = 0.0):
        """
        Description: Initializes model parameters

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
        assert self.initialized, "ERROR: Model not initialized!"
        
        # get prediction
        pred = np.dot(self.w, x)

        # update 
        self.w += self.Pinv @ x * (y - pred) / (1 + x.T @ self.Pinv @ x)
        self.Pinv -= (self.Pinv @ x) @ (self.Pinv @ x).T / (1 + x.T @ self.Pinv @ x)

        return pred

    def help(self):
        """
        Description: Prints information about this class and its methods.

        Args:
            None
        Returns:
            None
        """
        print(LeastSquares_help)



# string to print when calling help() method
LeastSquares_help = """

-------------------- *** --------------------

Id: LeastSquares
Description: Implements online least squares.

Methods:

    initialize(x, y, reg)
        Description: Initializes model parameters

        Args:
            x (float/numpy.ndarray): Observation
            y (float): label
            reg (float): regularization factor

    step(x)
        Description: Predict next value given observation and update internal parameters based
                     on true label.

        Args:
            x (float/numpy.ndarray): Observation
            y (float): label
        Returns:
            Predicted value for the next time-step

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""