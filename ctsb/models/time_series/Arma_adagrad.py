import ctsb
"""import jax.numpy as np """
import numpy as np 
import matplotlib.pyplot as plt
from ctsb.models.optimizers.Adagrad import Adagrad
from ctsb.models.optimizers.losses import mse

class ArmaAdaGrad(ctsb.CustomModel):
    """
    Implements the equivalent of an AR(p) model - predicts a linear
    combination of the previous p observed values in a time-series
    """

    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.uses_regressors = False

    def initialize(self, p = 3, optimizer=None, optimizer_params_dict=None, loss=None):
        """
        Description:
            Initializes autoregressive model parameters
        Args:
            p (int): Length of history used for prediction
        """
        self.initialized = True
        self.past = np.zeros(p + 1)
        self.params = np.zeros(p + 1)
        self.order = p
        self.G = np.ones(p+1)
        self.t = 1
        self.max_norm = 1
        self.optimizer = optimizer(pred=self.predict, 
                                 loss=loss, 
                                 learning_rate=None, 
                                 params_dict={'G':self.G, 'past':self.past, 'max_norm':self.max_norm, 'order':self.order})

    def predict(self, x):
        """
        Description:
            Predict next value given present value
        Args:
            x (int/numpy.ndarray):  Value at current time-step
        Returns:
            Predicted value for the next time-step
        """
        assert self.initialized
        """ and predict"""
        return np.dot(self.params, self.past)

    def update(self, y, loss=None):
        ''' update internal parameters '''
        self.params = self.optimizer.update(self.params, None, y, {'past': self.past})

        """ update the indices of the past """
        temp = np.zeros(self.order + 1)
        temp[0:self.order] = self.past[1:self.order + 1]
        temp[self.order] = y
        self.past = temp

    '''
    def update(self, y, loss = None):
        """
        Description:
            Updates parameters based on correct value, loss and learning rate.
        Args:
            y (int/numpy.ndarray): True value at current time-step
            loss (function): specifies loss function to be used; defaults to MSE
            lr (float): specifies learning rate; defaults to 0.001.
        Returns:
            None

        """
        grad = (np.dot(self.params, self.past) - y) * self.past
        if np.linalg.norm(grad) > self.max_norm:
            self.max_norm = np.linalg.norm(grad)

        self.t = self.t + 1
        self.G = self.G + np.square(grad)
        print(self.G)
        print(grad / np.sqrt(self.G))
        print('shalom')
        print(self.params)
        self.params = self.params -  (grad / np.sqrt(self.G))
        if np.linalg.norm(self.params) > self.order:
            self.params = self.params / np.linalg.norm(self.params) 

        return'''


    def help(self):
        """
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print('shalom')
