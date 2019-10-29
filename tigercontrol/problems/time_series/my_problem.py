import tigercontrol
import os
import jax.numpy as np
import pandas as pd
from tigercontrol.utils.dataset_registry import dataset_to_problem, get_tigercontrol_dir
from tigercontrol.error import StepOutOfBounds
from tigercontrol.problems.time_series import TimeSeriesProblem

class MyProblem(TimeSeriesProblem):
    """
    Description: Make dataset into tigercontrol problem class.
    """

    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.has_regressors = False

    def initialize(self, file, X = None, y = None, T = 1):
        """
        Description: Initialize the problem class.
        Args:
            file (string or dataframe): datapath to dataset or dataframe object 
            X (list of strings): ids of columns to use as features
            y (list of strings): ids of columns to use as y values
        Returns:
            X (numpy.ndarray): First observation
            y (numpy.ndarray): First label
        """

        self.initialized = True
        self.T = 0
        self.X, self.y = dataset_to_problem(file, X, y, T) # get data
        self.max_T = self.y.shape[0]
        self.has_regressors = True

        try:
            ret = (float(self.X[0]), float(self.y[0])) # throws error when X, y are arrays
        except:
            ret = (self.X[0], self.y[0])
        return ret

    def step(self):
        """
        Description: Moves time forward by one month and returns the corresponding observation and label.
        Args:
            None
        Returns:
            X (numpy.ndarray): Next observation
            y (numpy.ndarray): Next label
        """
        assert self.initialized

        self.T += 1
        if self.T == self.max_T: 
            raise StepOutOfBounds("Number of steps exceeded length of dataset ({})".format(self.max_T))

        try:
            ret = (float(self.X[self.T]), float(self.y[self.T])) # throws error when X, y are arrays
        except:
            ret = (self.X[self.T], self.y[self.T])
        return ret

    def hidden(self):
        """
        Description: Return the timestep corresponding to the last (observation, label) pair returned.
        Args:
            None
        Returns:
            Current timestep
        """
        assert self.initialized

        return "Timestep: {} out of {}".format(self.T + 1, self.max_T)

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
        print(Custom_help)

    def __str__(self):
        return "<Custom Problem>"


# string to print when calling help() method
ENSO_help = """

-------------------- *** --------------------

Id: ENSO-v0

Description: Collection of monthly values of control indices useful for predicting
             La Nina/El Nino. More specifically, the user can choose any of pna, ea,
             wa, wp, eu, soi, esoi, nino12, nino34, nino4, oni of nino34 (useful for
             La Nino/El Nino identification) to be used as input and/or output in
             the problem instance.

Methods:

    initialize(input_signals, include_month, output_signals, history, timeline)
        Description:
            Initializes the ctrl_indices dataset to a format suited to the online learning setting.
            By default, the current values of all available signals are used to predict the next
            value of nino34's oni. 
        Args:
            input_signals (list of strings): signals used for prediction
            include_month (boolean): True if the month should be used as a feature,
                                     False otherwise
            output_signals (list of strings): signals we are trying to predict
            history (int): number of past observations used for prediction
            timeline (int/list of ints): the forecasting timeline(s)
        Returns:
            X (numpy.ndarray): First observation
            y (numpy.ndarray): First label

    step()
        Description:
            Moves time forward by one month and returns the corresponding observation and label.
        Args:
            None
        Returns:
            X (numpy.ndarray): Next observation
            y (numpy.ndarray): Next label

    hidden()
        Description:
            Return the timestep corresponding to the last (observation, label) pair returned.
        Args:
            None
        Returns:
            Current timestep

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""


