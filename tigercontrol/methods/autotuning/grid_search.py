"""
Hyperparameter tuning using (optionally random) Grid Search.
"""

import tigercontrol
import jax
import jax.numpy as np

class GridSearch:
    """
    Description: Implements the equivalent of an AR(p) method - predicts a linear
    combination of the previous p observed values in a time-series
    """


    def __init__(self):
        self.initialized = False

    def initialize(self, method_id, method_params, problem_id, problem_params, search_space, trials=None):
        """
        Description: Initializes autoregressive method parameters
        Args:
            method_id (string): id of method
            method_params (dict): initial method parameters dict (updated by search space)
            problem_id (string): id of problem to try on
            problem_params (dict): problem parameters dict
            search_space (string): dict mapping parameter names to a finite set of options
            trials (int, None): number of random trials to sample from search space / try all parameters
        """
        return



# string to print when calling help() method
GridSearch_help = """

-------------------- *** --------------------

Id: AutoRegressor
Description: Implements the equivalent of an AR(p) method - predicts a linear
    combination of the previous p observed values in a time-series

Methods:

    initialize()
        Description:
            Initializes autoregressive method parameters
        Args:
            p (int): Length of history used for prediction

    step(x)
        Description:
            Run one timestep of the method in its environment then update internal parameters
        Args:
            x (int/numpy.ndarray):  Value at current time-step
        Returns:
            Predicted value for the next time-step

    predict(x)
        Description:
            Predict next value given present value
        Args:
            x (int/numpy.ndarray):  Value at current time-step
        Returns:
            Predicted value for the next time-step

    update(y, loss, lr)
        Description:
            Updates parameters based on correct value, loss and learning rate.
        Args:
            y (int/numpy.ndarray): True value at current time-step
            loss (function): (optional)
            lr (float):
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