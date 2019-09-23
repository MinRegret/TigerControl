"""
bitcoin daily price
"""

import tigercontrol
import os
import jax.numpy as np
import pandas as pd
from datetime import datetime
from tigercontrol.utils import crypto, get_tigercontrol_dir
from tigercontrol.error import StepOutOfBounds
from tigercontrol.problems.time_series import TimeSeriesProblem


class Crypto(TimeSeriesProblem):
    """
    Description: Outputs the daily price of bitcoin from 2013-04-28 to 2018-02-10
    """
    
    compatibles = set(['Crypto-v0', 'TimeSeries'])

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
        self.df = crypto() # get data
        self.max_T = self.df.shape[0]
        
        return self.df.iloc[self.T, 3]

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
        return self.df['Price'].iloc[self.T]

    def hidden(self):
        """
        Description: Return the date corresponding to the last price of bitcoin that was returned
        Args:
            None
        Returns:
            Date (string)
        """
        assert self.initialized
        ts = self.df['Date'].iloc[self.T]/1000
        date_time = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        return "Timestep: {} out of {}".format(self.T+1, self.max_T) + '\n' + str(date_time)

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
        print(crypto_help)

    def __str__(self):
        return "<Crypto Problem>"


# string to print when calling help() method
crypto_help = """

-------------------- *** --------------------

Id: Crypto-v0
Description: Outputs the daily price of bitcoin
        from 2013-04-28 to 2018-02-10

Methods:

    initialize()
            Check if data exists, else download, clean, and setup.
        Args:
            None
        Returns:
            The first bitcoin price

    step()
        Description:
            Moves time forward by one day and returns price of the bitcoin
        Args:
            None
        Returns:
            The next bitcoin price

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


