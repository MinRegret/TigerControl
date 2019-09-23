"""
Produces randomly generated scalar values at every timestep, taken from a normal distribution.
"""
import jax.numpy as np
import jax.random as random
import tigercontrol
from tigercontrol.utils.random import generate_key
from tigercontrol.problems.time_series import TimeSeriesProblem

class Random(TimeSeriesProblem):
    """
    Description: A random sequence of scalar values taken from an i.i.d. normal distribution.
    """

    compatibles = set(['Random-v0', 'TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.has_regressors = False

    def initialize(self):
        """
        Description: Randomly initialize the hidden dynamics of the system.
        Args:
            None
        Returns:
            None
        """
        self.T = 0
        self.initialized = True
        return random.normal(generate_key())

    def step(self):
        """
        Description: Moves the system dynamics one time-step forward.
        Args:
            None
        Returns:
            The next value in the time-series.
        """
        assert self.initialized
        self.T += 1
        return random.normal(generate_key())

    def hidden(self):
        """
        Not implemented
        """
        pass

    def close(self):
        """
        Not implemented
        """
        pass

    def help(self):
        """
        Description: Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(Random_help)



# string to print when calling help() method
Random_help = """

-------------------- *** --------------------

Id: Random-v0
Description: A random sequence of scalar values taken from an i.i.d. normal distribution.

Methods:

    initialize()
        Description:
            Randomly initialize the hidden dynamics of the system.
        Args:
            None
        Returns:
            None

    step()
        Description:
            Moves the system dynamics one time-step forward.
        Args:
            None
        Returns:
            The next value in the time-series.

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""