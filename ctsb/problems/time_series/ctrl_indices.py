"""
S&P 500 daily opening price
"""

import ctsb
import os
import jax.numpy as np
import pandas as pd
from ctsb.utils.dataset_registry import ctrl_indices, get_ctsb_dir
from ctsb.error import StepOutOfBounds
from ctsb.problems.time_series import TimeSeriesProblem

class CtrlIndices(TimeSeriesProblem):
    """
    Description: ...
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, input_signals = ['pna', 'ea', 'wa', 'wp', 'eu', 'soi', 'esoi', 'nino12', 'nino34', 'nino4'], include_month = False, output_signals = ['oni'], history = 1, timeline = 1):
        """
        Description:
            Check if data exists, else download, clean, and setup.
        Args:
            None
        Returns:
            The first tuple of observations and corresponding label
        """
        self.initialized = True
        self.T = 0
        self.X, self.y = ctrl_indices(input_signals, include_month, output_signals, history, timeline) # get data
        self.max_T = self.y.shape[0]

        return (self.X[0], self.y[0])

    def step(self):
        """
        Description:
            Moves time forward by one day and returns value of the stock index
        Args:
            None
        Returns:
            The next tuple of observations and corresponding label
        """
        assert self.initialized

        self.T += 1
        if self.T == self.max_T: 
            raise StepOutOfBounds("Number of steps exceeded length of dataset ({})".format(self.max_T))

        return (self.X[self.T], self.y[self.T])

    def hidden(self):
        """
        Description:
            Return the date corresponding to the last value of the S&P 500 that was returned
        Args:
            None
        Returns:
            Date (string)
        """
        assert self.initialized

        return "Timestep: {} out of {}".format(self.T+1, self.max_T)

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
        print(ARMA_help)

    def __str__(self):
        return "<SP500 Problem>"


# string to print when calling help() method
ARMA_help = """

-------------------- *** --------------------

Id: SP500-v0
Description: Outputs the daily opening price of the S&P 500 stock market index from
    January 3, 1986 to June 29, 2018.

Methods:

    initialize()
            Check if data exists, else download, clean, and setup.
        Args:
            None
        Returns:
            The first S&P 500 value

    step()
        Description:
            Moves time forward by one day and returns value of the stock index
        Args:
            None
        Returns:
            The next S&P 500 value

    hidden()
        Description:
            Return the date corresponding to the last value of the S&P 500 that was returned
        Args:
            None
        Returns:
            Date (string)

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""


