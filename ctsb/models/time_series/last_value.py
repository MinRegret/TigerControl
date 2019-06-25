"""
Last observed value
"""

import jax.numpy as np
import ctsb
from ctsb.utils import seeding

class LastValue(ctsb.Model):
    """
    Predicts the last value in the time series, i.e. x_t = x_(t-1)
    """

    def __init__(self):
        self.initialized = False

    def initialize(self):
        """
        Description:
            Initialize the (non-existent) hidden dynamics of the model.
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
        return x

    def predict(self, x):
        """
        Description:
            Takes input observation and returns next prediction value
        Args:
            x (float/numpy.ndarray): value at current time-step
        Returns:
            Predicted value for the next time-step
        """
        return x

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
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(LastValue_help)



# string to print when calling help() method
LastValue_help = """

-------------------- *** --------------------

Id: LastValue
Description: Predicts the last value in the time series, i.e. x_t = x_(t-1)

Methods:

    initialize()
        Description:
            Initialize the (non-existent) hidden dynamics of the model.
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
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""