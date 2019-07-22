"""
Last observed value
"""

import jax.numpy as np
import ctsb
from ctsb.models.time_series import TimeSeriesModel

class LastValue(TimeSeriesModel):
    """
    Description: Predicts the last value in the time series, i.e. x(t) = x(t-1)
    """

    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.uses_regressors = False

    def initialize(self):
        """
        Description: Initialize the (non-existent) hidden dynamics of the model
        Args:
            None
        Returns:
            None
        """
        self.initialized = True
        self.x = 0

    def predict(self, x):
        """
        Description: Takes input observation and returns next prediction value
        Args:
            x (float/numpy.ndarray): value at current time-step
        Returns:
            Predicted value for the next time-step
        """
        return self.x

    def update(self, y):
        """
        Description: Takes update rule and adjusts internal parameters
        Args:
            rule (function): rule with which to alter parameters
        Returns:
            None
        """
        self.x = y

    def help(self):
        """
        Description: Prints information about this class and its methods
        Args:
            None
        Returns:
            None
        """
        print(LastValue_help)

    def __str__(self):
        return "<LastValue Model>"



# string to print when calling help() method
LastValue_help = """

-------------------- *** --------------------

Id: LastValue
Description: Predicts the last value in the time series, i.e. x(t) = x(t-1)

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