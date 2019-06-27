"""
Linear combination of previous values
"""

import ctsb
import jax.numpy as np

class Linear(ctsb.Model):
    """
    Implements the equivalent of an AR(p) model - predicts a linear combination of the
    previous p observed values in a time-series
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, p):
        # initializes model parameters
        self.initialized = True
        raise NotImplementedError

    def step(self, **kwargs):
        # run one timestep of the model in its environment
        raise NotImplementedError

    def predict(self, x=None):
        # returns model prediction for given input
        raise NotImplementedError

    def update(self, rule=None):
        # update parameters according to given loss and update rule
        raise NotImplementedError

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