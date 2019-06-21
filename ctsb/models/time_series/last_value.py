"""
Last observed value
"""

import ctsb
import jax.numpy as np
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
            Randomly initialize the hidden dynamics of the system.
        Args:
            None
        Returns:
            None
        """
        self.initialized = True

    def step(self):
        """
        Description:
            Moves the system dynamics one time-step forward.
        Args:
            None
        Returns:
            The next value in the time-series.
        """
        assert self.initialized
        return np.random.normal()

    def hidden(self):
        """
        Not implemented
        """
        pass

    def seed(self, seed=None):
        """
        Description:
            Seeds the random number generator to produce deterministic, reproducible results. 
        Args:
            seed (int): Default value None. The number that determines the seed of the random
            number generator for the system.
        Returns:
            A list containing the resulting NumPy seed.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        """
        Not implemented
        """
        pass

    def help(self):
        """
        Description:
            Prints information about this class and its methods.
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

    seed(seed)
        Description:
            Seeds the random number generator to produce deterministic, reproducible results. 
        Args:
            seed (int): Default value None. The number that determines the seed of the random
            number generator for the system.
        Returns:
            A list containing the resulting NumPy seed.

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""