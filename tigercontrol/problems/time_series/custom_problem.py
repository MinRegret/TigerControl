import tigercontrol
import os
import jax.numpy as np
import pandas as pd
from tigercontrol.utils.dataset_registry import dataset_to_problem, get_tigercontrol_dir
from tigercontrol.error import StepOutOfBounds
from tigercontrol.problems.time_series import TimeSeriesProblem

class MyProblem(TimeSeriesProblem):
    """
    Description: 
    """

    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.has_regressors = False

    def initialize(self, file, X = None, y = None):
        """
        Description: 
        Args:
        Returns:
            X (numpy.ndarray): First observation
            y (numpy.ndarray): First label
        """

        self.initialized = True
        self.T = 0
        self.X, self.y = dataset_to_problem(file, X, y) # get data
        self.max_T = self.y.shape[0]

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

    def __str__(self):
        return "<Custom Problem>"

