"""
Last observed value
"""

import jax.numpy as np
import jax.random as rand
import ctsb
from ctsb.models.time_series import TimeSeriesModel
from ctsb.utils import generate_key

class WaveFiltering(TimeSeriesModel):
    """
    Description: Predicts the last value in the time series, i.e. x(t) = x(t-1)
    """

    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.uses_regressors = False

    def initialize(self, n, m, k, T, eta, R_M, k_values, k_vectors):
        """
        Description: Initialize the (non-existent) hidden dynamics of the model
        Args:
            None
        Returns:
            None
        """
        self.initialized = True
        self.n = n
        self.m = m
        self.k = k
        self.T = T
        self.eta = eta
        self.R_M = R_M
        self.k_values = k_values
        self.k_vectors = k_vectors
        self.k_prime = n * k + 2 * n + m
        self.M = 2 * rand.uniform(generate_key(), shape=(self.m, self.k_prime)) - 1
        if (4 * k > T):
            raise Exception("Model parameter k must be less than T/4")
        self.X = np.array([])
        self.Y = np.array([])
        self.X_sim = None
        self.t = 0
        self.y_hat = None

    def predict(self, x):
        """
        Description: Takes input observation and returns next prediction value
        Args:
            x (float/numpy.ndarray): value at current time-step
        Returns:
            Predicted value for the next time-step
        """
        if self.X.size == 0:
            self.X = np.asarray([x]).T
        else:
            self.X = np.hstack((self.X, np.asarray([x]).T))

        if (self.t == 0): # t = 0 results in an excessively complicated corner case otherwise
            self.X_sim = np.append(np.zeros(self.n * self.k + self.n), np.append(self.X[:,0], np.zeros(self.m)))
        else:
            eigen_diag = np.diag(self.k_values**0.25)
            if (self.t <= self.T):
                X_sim_pre = self.X[:,0:self.t-1].dot(np.flipud(self.k_vectors[0:self.t-1,:])).dot(eigen_diag)
            else:
                X_sim_pre = self.X[:,self.t-self.T-1:self.t-1].dot(np.flipud(self.k_vectors)).dot(eigen_diag)
            x_y_cols = np.append(np.append(self.X[:,self.t-1], self.X[:,self.t]), self.Y[:,self.t-1])
            self.X_sim = np.append(X_sim_pre.T.flatten(), x_y_cols)
        self.y_hat = self.M.dot(self.X_sim)
        return self.y_hat

    def forecast(self, x, timeline = 1):
        """
        Description: Forecast values 'timeline' timesteps in the future
        Args:
            x (int/numpy.ndarray):  Value at current time-step
            timeline (int): timeline for forecast
        Returns:
            Forecasted values 'timeline' timesteps in the future
        """
        return np.ones(timeline) * x

    def update(self, y):
        """
        Description: Takes update rule and adjusts internal parameters
        Args:
            y (float/np.ndarray): true value
        Returns:
            None
        """
        if self.Y.size == 0:
            self.Y = np.asarray([y]).T
        else:
            # self.Y = np.append(self.Y, np.asarray([y]).T)
            self.Y = np.hstack((self.Y, np.asarray([y]).T))
        y_delta = np.asarray([self.y_hat]).T - np.asarray([y]).T
        self.M = self.M - 2 * self.eta * np.outer(y_delta, self.X_sim) # changed from +2 to -2
        if (np.linalg.norm(self.M) > self.R_M):
            self.M = self.M * (self.R_M / np.linalg.norm(self.M))
        self.t += 1
        return

    def help(self):
        """
        Description: Prints information about this class and its methods
        Args:
            None
        Returns:
            None
        """
        print(LastValue_help)

    def __str__(self):
        return "<WaveFiltering Model>"



# string to print when calling help() method
WaveFiltering_help = """

-------------------- *** --------------------

Id: LastValue
Description: Predicts the last value in the time series, i.e. x(t) = x(t-1)

Methods:

    initialize()
        Description:
            Initialize the (non-existent) hidden dynamics of the model
        Args:
            None
        Returns:
            None

    step(x)
        Description:
            Takes input observation and returns next prediction value,
            then updates internal parameters
        Args:
            x (float/numpy.ndarray): value at current time-step
        Returns:
            Predicted value for the next time-step

    predict(x)
        Description:
            Takes input observation and returns next prediction value
        Args:
            x (float/numpy.ndarray): value at current time-step
        Returns:
            Predicted value for the next time-step

    update(rule=None)
        Description:
            Takes update rule and adjusts internal parameters
        Args:
            rule (function): rule with which to alter parameters
        Returns:
            None

    help()
        Description:
            Prints information about this class and its methods
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""