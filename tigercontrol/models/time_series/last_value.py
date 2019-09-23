"""
Last observed value
"""

import jax.numpy as np
import tigercontrol
from tigercontrol.models.time_series import TimeSeriesModel

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

    def predict(self, x):
        """
        Description: Takes input observation and returns next prediction value
        Args:
            x (float/numpy.ndarray): value at current time-step
        Returns:
            Predicted value for the next time-step
        """
        return x

    def forecast(self, x, timeline = 1):
        """
        Description: Forecast values 'timeline' timesteps in the future
        Args:
            x (int/numpy.ndarray):  Value at current time-step
            timeline (int): timeline for forecast
        Returns:
            Forecasted values 'timeline' timesteps in the future
        """
        return np.ones(timeline) * x

    def update(self, y):
        """
        Description: Takes update rule and adjusts internal parameters
        Args:
            y (float/np.ndarray): true value
        Returns:
            None
        """
        return

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