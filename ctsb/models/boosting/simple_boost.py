"""
AR(p): Linear combination of previous values
"""

import ctsb
import jax
import jax.numpy as np
from ctsb.models.optimizers.losses import mse

class SimpleBoost:
    """
    Description: Implements the equivalent of an AR(p) model - predicts a linear
    combination of the previous p observed values in a time-series
    """

    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False

    def initialize(self, model_id, model_params, N=10, loss=mse, reg=1.0):
        """
        Description: Initializes autoregressive model parameters
        Args:
            model_id (string): id of weak learner model
            model_params (dict): dict of params to pass model
            N (int): default 3. Number of weak learners
            loss (function): loss function for boosting method
            reg (float): default 1.0. constant for regularization.
        """
        self.initialized = True

        # initialize proxy loss
        proxy_loss = lambda y_pred, v: np.dot(y_pred, v) + (reg/2) * np.sum(y_pred**2)

        # 1. Maintain N copies of the algorithm 
        assert N > 0
        self.N = N
        self.models = []
        for _ in range(N):
            new_model = ctsb.model(model_id)
            new_model.initialize(**model_params)
            new_model.optimizer.set_loss(proxy_loss) # proxy loss
            self.models.append(new_model)

        def _prev_predict(x):
            y = []
            cur_y = 0
            for i, model_i in enumerate(self.models):
                eta_i = 2 / (i + 2)
                y_pred = model_i.predict(x)
                cur_y = (1 - eta_i) * cur_y + eta_i * y_pred
                y.append(cur_y)
            return [np.zeros(shape=y[0].shape)] + y
        self._prev_predict = _prev_predict

        def _get_grads(y_true, prev_predicts):
            g = jax.grad(loss)
            v_list = [g(y_prev, y_true) for y_prev in prev_predicts]
            return v_list
        self._get_grads = jax.jit(_get_grads)


    def to_ndarray(self, x):
        """
        Description: If x is a scalar, transform it to a (1, 1) numpy.ndarray;
        otherwise, leave it unchanged.
        Args:
            x (float/numpy.ndarray)
        Returns:
            A numpy.ndarray representation of x
        """
        x = np.asarray(x)
        if np.ndim(x) == 0:
            x = x[None]
        return x


    def predict(self, x):
        assert self.initialized
        x = self.to_ndarray(x)
        self._prev_predicts = self._prev_predict(x)
        return self._prev_predicts[-1]


    def update(self, y):
        assert self.initialized
        grads = self._get_grads(y, self._prev_predicts)
        for grad_i, model_i in zip(grads[:-1], self.models):
            model_i.update(grad_i)


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