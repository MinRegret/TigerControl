"""
AR(p): Linear combination of previous values
"""

import ctsb
import jax
import jax.numpy as np
from ctsb.models.time_series import TimeSeriesModel
from ctsb.models.optimizers import SGD, OGD
from ctsb.models.optimizers.losses import mse, boost_mse

class SimpleBoost(object):
    """
    Description: Implements the equivalent of an AR(p) model - predicts a linear
    combination of the previous p observed values in a time-series
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, model, N = 10):
        """
        Description: Initializes autoregressive model parameters
        Args:
            p (int): Length of history used for prediction
            optimizer (class): optimizer choice
            loss (class): loss choice
            lr (float): learning rate for update
        """
        (model_id, model_params) = model

        optimizer = OGD()

        # 1. Maintain N copies of the algorithm 
        self.N = N
        self.models = []
        for _ in range(N):
            new_model = ctsb.model(model_id)
            new_model.initialize()
            self.models.append(new_model)

        def _predict(x):
            y = []
            cur_y = 0
            for i in range(N):
                eta_i = 2 / (i + 2)
                cur_y = (1 - eta_i) * cur_y + eta_i * models[i].predict(x)
                y.append(cur_y)
            return y

        ''' 2. Ensure they all have different losses '''
        for i in range(N):

            def _loss(y_pred, y_true):
                return 1. / L_D * np.dot(2 * y_pred[i] - 2 * y_true, y_true)

            self.models[i]._store_optimizer(optimizer, _predict)
            self.models[i].optimizer.set_loss(_loss)

    def predict(self, x):
        return self._predict(x)[self.N - 1]

    def update(self, y):
        for i in range(N):
            self.models[i].update(y)

    def help(self):
        """
        Description: Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(AutoRegressor_help)



# string to print when calling help() method
AutoRegressor_help = """

-------------------- *** --------------------

Id: AutoRegressor
Description: Implements the equivalent of an AR(p) model - predicts a linear
    combination of the previous p observed values in a time-series

Methods:

    initialize()
        Description:
            Initializes autoregressive model parameters
        Args:
            p (int): Length of history used for prediction

    step(x)
        Description:
            Run one timestep of the model in its environment then update internal parameters
        Args:
            x (int/numpy.ndarray):  Value at current time-step
        Returns:
            Predicted value for the next time-step

    predict(x)
        Description:
            Predict next value given present value
        Args:
            x (int/numpy.ndarray):  Value at current time-step
        Returns:
            Predicted value for the next time-step

    update(y, loss, lr)
        Description:
            Updates parameters based on correct value, loss and learning rate.
        Args:
            y (int/numpy.ndarray): True value at current time-step
            loss (function): (optional)
            lr (float):
        Returns:
            None

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""