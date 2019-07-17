"""
Recurrent neural network output
"""
import jax
import jax.numpy as np
import jax.experimental.stax as stax
import ctsb
from ctsb.utils.random import generate_key
from ctsb.models.time_series import TimeSeriesModel
from ctsb.models.optimizers.SGD import SGD
from ctsb.models.optimizers.losses import mse

class RNN(TimeSeriesModel):
    """
    Produces outputs from a randomly initialized recurrent neural network.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, n, m, l=32, h=64, optimizer=None):
        """
        Description:
            Randomly initialize the RNN.
        Args:
            n (int): Input dimension.
            m (int): Observation/output dimension.
            l (int): Length of memory for update step purposes.
            h (int): Default value 64. Hidden dimension of RNN.
            update (func): update function implemented with Jax NumPy,
                takes params, pred, x, y as input and returns updated params
        Returns:
            The first value in the time-series
        """
        self.T = 0
        self.initialized = True
        self.n, self.m, self.l, self.h = n, m, l, h

        # initialize parameters
        glorot_init = stax.glorot() # returns a function that initializes weights
        W_h = glorot_init(generate_key(), (h, h))
        W_x = glorot_init(generate_key(), (h, n))
        W_out = glorot_init(generate_key(), (m, h))
        b_h = np.zeros(h)
        self.params = [W_h, W_x, W_out, b_h]
        self.hid = np.zeros(h)
        self.x = np.zeros((l, n))

        # initialize jax.jitted predict and update functions

        def _fast_predict(params, x, hid):
            W_h, W_x, W_out, b_h = params
            next_hid = np.tanh(np.dot(W_h, hid) + np.dot(W_x, x) + b_h)
            y = np.dot(W_out, next_hid)
            return (y, next_hid)
        self._fast_predict = jax.jit(_fast_predict)

        def _slow_predict(params, x_list):
            W_h, W_x, W_out, b_h = params
            next_hid = np.zeros(self.h)
            for x in x_list:
                next_hid = np.tanh(np.dot(W_h, next_hid) + np.dot(W_x, x) + b_h)
            y = np.dot(W_out, next_hid)
            return y
        self._slow_predict = jax.jit(_slow_predict)

        def _update_x(self_x, x):
            new_x = np.roll(self_x, self.n)
            new_x = jax.ops.index_update(new_x, jax.ops.index[0,:], x)
            return new_x
        self._update_x = jax.jit(_update_x)

        '''
        if update:
            self._update = jax.jit(update)
        else:
            def _update(params, pred, x, y_true):
                lr = 0.01 # learning rate
                def _loss(params, x, y_true):
                    y_pred = pred(params, x)
                    loss = np.sum((y_pred - y_true)**2)
                    return loss
                _gradient = jax.grad(_loss)
                delta = _gradient(params, x, y_true)
                new_params = [w - lr*dw for w, dw in zip(params, delta)]
                return new_params
            self._update = jax.jit(_update, static_argnums=[1])
        '''
        self.optimizer = SGD(pred=self._slow_predict, loss=mse, learning_rate=0.003)
        return

    def predict(self, x):
        self.x = self._update_x(self.x, x)
        y, self.hid = self._fast_predict(self.params, x, self.hid)
        return y

    # def update(self, y, loss=None):
    #     self.params = self._update(self.params, self._slow_predict, self.x, y)
    #    return

    def update(self, y):
        self.params = self.optimizer.update(self.x, y, self.params)
        return

    def help(self):
        """
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(RNN_help)



# string to print when calling help() method
RNN_help = """

-------------------- *** --------------------

Id: RNN
Description: Implements a Recurrent Neural Network model.

Methods:

    initialize(n, m, l=32, h=64, update=None):
        Description:
            Randomly initialize the RNN.
        Args:
            n (int): Input dimension.
            m (int): Observation/output dimension.
            l (int): Length of memory for update step purposes.
            h (int): Default value 64. Hidden dimension of RNN.
            update (func): update function implemented with Jax NumPy,
                takes params, pred, x, y as input and returns updated params

    predict(x)
        Description:
            Makes a prediction given input x
        Args:
            x (numpy.ndarray): RNN input, an n-dimensional real-valued vector.
        Returns:
            The output of the RNN computed on the past l inputs, including the new x.

    update(y)
        Description:
            Updates internal parameters
        Args:
            None
        Returns:
            h: The hidden state.

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""







