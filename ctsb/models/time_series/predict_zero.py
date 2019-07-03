"""
Predict 0
"""

import jax.numpy as np
import ctsb
from ctsb.models.time_series import TimeSeriesModel

class PredictZero(TimeSeriesModel):
    """
    Predicts the next value in the time series to be 0, i.e. x(t) = 0
    """

    def __init__(self):
        self.initialized = False

    def initialize(self):
        """
        Description:
            Initialize the (non-existent) hidden dynamics of the model
        Args:
            None
        Returns:
            None
        """
        self.initialized = True

    def step(self, x):
        """
        Description:
            Takes input observation and returns next prediction value,
            then updates internal parameters
        Args:
            x (float/numpy.ndarray): value at current time-step
        Returns:
            Predicted value for the next time-step
        """
        return 0

    def predict(self, x):
        """
        Description:
            Takes input observation and returns next prediction value
        Args:
            x (float/numpy.ndarray): value at current time-step
        Returns:
            Predicted value for the next time-step
        """
        return 0

    def update(self, rule=None):
        """
        Description:
            Takes update rule and adjusts internal parameters
        Args:
            rule (function): rule with which to alter parameters
        Returns:
            None
        """
        return

    def help(self):
        """
        Description:
            Prints information about this class and its methods
        Args:
            None
        Returns:
            None
        """
        print(PredictZero_help)

    def __str__(self):
        return "<PredictZero Model>"



# string to print when calling help() method
PredictZero_help = """

-------------------- *** --------------------

Id: PredictZero
Description: Predicts the next value in the time series to be 0, i.e. x(t) = 0

Methods:

    initialize()
        Description:
            Initialize the (non-existent) hidden dynamics of the model
        Args:
            None
        Returns:
            None

    step(x)
        Description:
            Takes input observation and returns next prediction value,
            then updates internal parameters
        Args:
            x (float/numpy.ndarray): value at current time-step
        Returns:
            Predicted value for the next time-step

    predict(x)
        Description:
            Takes input observation and returns next prediction value
        Args:
            x (float/numpy.ndarray): value at current time-step
        Returns:
            Predicted value for the next time-step

    update(rule=None)
        Description:
            Takes update rule and adjusts internal parameters
        Args:
            rule (function): rule with which to alter parameters
        Returns:
            None

    help()
        Description:
            Prints information about this class and its methods
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""