'''
SGD optimizer
'''
from tigerforecast.methods.optimizers.core import Optimizer
from tigerforecast.methods.optimizers.losses import mse

class SGD(Optimizer):
    """
    Description: Stochastic Gradient Descent optimizer.
    Args:
        pred (function): a prediction function implemented with jax.numpy 
        loss (function): specifies loss function to be used; defaults to MSE
        learning_rate (float): learning rate
    Returns:
        None
    """
    def __init__(self, pred=None, loss=mse, learning_rate=0.0001, hyperparameters={}):
        self.initialized = False
        self.lr = learning_rate
        self.hyperparameters = hyperparameters
        self.pred = pred
        self.loss = loss
        if self._is_valid_pred(pred, raise_error=False) and self._is_valid_loss(loss, raise_error=False):
            self.set_predict(pred, loss=loss)

    def reset(self): # mandatory for every function, but SGD doesn't make any changes
        pass

    def update(self, params, x, y, loss=None):
        """
        Description: Updates parameters based on correct value, loss and learning rate.
        Args:
            params (list/numpy.ndarray): Parameters of method pred method
            x (float): input to method
            y (float): true label
            loss (function): loss function. defaults to input value.
        Returns:
            Updated parameters in same shape as input
        """
        assert self.initialized
        grad = self.gradient(params, x, y, loss=loss) # defined in optimizers core class
        if (type(params) is list):
            return [w - self.lr * dw for (w, dw) in zip(params, grad)]
        return params - self.lr * grad

    def __str__(self):
        return "<SGD Optimizer, lr={}>".format(self.lr)