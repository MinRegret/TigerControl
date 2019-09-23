"""
monthly unemployment for the past few decades
"""

import tigercontrol
import os
import jax.numpy as np
import pandas as pd
from datetime import datetime
from tigercontrol.utils import unemployment, get_tigercontrol_dir
from tigercontrol.error import StepOutOfBounds
from tigercontrol.problems.time_series import TimeSeriesProblem


class Unemployment(TimeSeriesProblem):
    """
    Description: Monthly unemployment rate since 1948.
    """

    compatibles = set(['Unemployment-v0', 'TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.has_regressors = False

    def initialize(self):
        """
        Description: Check if data exists, else download, clean, and setup.
        Args:
            None
        Returns:
            The first bitcoin price
        """
        self.initialized = True
        self.T = 0
        self.df = unemployment() # get data
        self.max_T = self.df.shape[0]
        
        return self.df.iloc[self.T, 1]

    def step(self):
        """
        Description: Moves time forward by one day and returns price of the bitcoin
        Args:
            None
        Returns:
            The next bitcoin price
        """
        assert self.initialized
        self.T += 1
        if self.T == self.max_T: 
            raise StepOutOfBounds("Number of steps exceeded length of dataset ({})".format(self.max_T))
        return self.df.iloc[self.T, 1]

    def hidden(self):
        """
        Description: Return the date corresponding to the last unemployment rate value
        Args:
            None
        Returns:
            Date (string)
        """
        assert self.initialized
        return self.df.iloc[self.T, 0]

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
        print(Unemployment_help)

    def __str__(self):
        return "<Unemployment Problem>"


# string to print when calling help() method
Unemployment_help = """

-------------------- *** --------------------

Id: Unemployment-v0
Description: Outputs monthly unemployment rate from 1948 until today.

Methods:

    initialize()
            Check if data exists, else download, clean, and setup.
        Args:
            None
        Returns:
            The first unemployment rate

    step()
        Description:
            Moves time forward by one day and returns price of the bitcoin
        Args:
            None
        Returns:
            Next unemployment rate

    hidden()
        Description:
            Return the date corresponding to the last price of bitcoin that was returned
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


