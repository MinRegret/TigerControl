"""
UCI indoor temperature data
"""

import tigercontrol
import os
import jax.numpy as np
import pandas as pd
from tigercontrol.utils import uci_indoor, get_tigercontrol_dir
from tigercontrol.error import StepOutOfBounds
from tigercontrol.problems.time_series import TimeSeriesProblem

class UCI_Indoor(TimeSeriesProblem):
    """
    Description: Outputs various weather metrics from a UCI dataset from 13/3/2012 to 11/4/2012
    """

    compatibles = set(['UCI-Indoor-v0', 'TimeSeries'])

    def __init__(self):
        self.initialized = False
        # self.data_path = os.path.join(get_tigercontrol_dir(), "data/uci_indoor_cleaned.csv")
        self.pred_indices = []
        self.has_regressors = False

    def initialize(self, pred_indices=['5:Weather_Temperature']):
        """
        Description: Check if data exists, else download, clean, and setup.
        Args:
            None
        Returns:
            The first uci_indoor value
        """
        self.initialized = True
        self.T = 0
        self.df_full = uci_indoor() # get data
        self.df = self.df_full.drop(['1:Date','2:Time'],axis=1)
        self.max_T = self.df.shape[0]
        self.pred_indices = pred_indices
        
        return np.array(self.df.iloc[self.T]['7:CO2_Habitacion_Sensor'])
        # return (self.df.iloc[self.T].drop(self.pred_indices).as_matrix(), self.df.iloc[self.T][self.pred_indices].as_matrix())

    def step(self):
        """
        Description: Moves time forward by fifteen minutes and returns weather metrics
        Args:
            None
        Returns:
            The next uci_indoor value
        """
        assert self.initialized
        self.T += 1
        if self.T == self.max_T: 
            raise StepOutOfBounds("Number of steps exceeded length of dataset ({})".format(self.max_T))
        # return (self.df.iloc[self.T].drop(self.pred_indices).as_matrix(), self.df.iloc[self.T][self.pred_indices].as_matrix())
        return np.array(self.df.iloc[self.T]['7:CO2_Habitacion_Sensor'])

    def hidden(self):
        """
        Description: Return the date corresponding to the last value of the uci_indoor that was returned
        Args:
            None
        Returns:
            Date (string)
        """
        assert self.initialized
        return "Timestep: {} out of {}".format(self.T+1, self.max_T) + '\n' + str(self.df_full.iloc[self.T][['1:Date','2:Time','24:Day_Of_Week']])

    def close(self):
        """
        Not implemented
        """
        pass
