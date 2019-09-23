"""
S&P 500 daily opening price
"""

import tigercontrol
import os
import jax.numpy as np
import pandas as pd
from tigercontrol.utils import sp500, get_tigercontrol_dir
from tigercontrol.error import StepOutOfBounds
from tigercontrol.problems.time_series import TimeSeriesProblem

class SP500(TimeSeriesProblem):
    """
    Description: Outputs the daily opening price of the S&P 500 stock market index 
    from January 3, 1986 to June 29, 2018.
    """

    compatibles = set(['SP500-v0', 'TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.data_path = os.path.join(get_tigercontrol_dir(), "data/sp500.csv")
        self.has_regressors = False

    def initialize(self):
        """
        Description: Check if data exists, else download, clean, and setup.
        Args:
            None
        Returns:
            The first S&P 500 value
        """
        self.initialized = True
        self.T = 0
        self.df = sp500() # get data
        self.max_T = self.df.shape[0]
        self.has_regressors = False
        
        return self.df.iloc[self.T, 1]

    def step(self):
        """
        Description: Moves time forward by one day and returns value of the stock index
        Args:
            None
        Returns:
            The next S&P 500 value
        """
        assert self.initialized
        self.T += 1
        if self.T == self.max_T: 
            raise StepOutOfBounds("Number of steps exceeded length of dataset ({})".format(self.max_T))
        return self.df.iloc[self.T, 1]

    def hidden(self):
        """
        Description: Return the date corresponding to the last value of the S&P 500 that was returned
        Args:
            None
        Returns:
            Date (string)
        """
        assert self.initialized
        return "Timestep: {} out of {}, date: ".format(self.T+1, self.max_T) + self.df.iloc[self.T, 0]

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
        print(SP500_help)

    def __str__(self):
        return "<SP500 Problem>"


# string to print when calling help() method
SP500_help = """

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


