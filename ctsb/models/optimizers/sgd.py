'''
SGD optimizer
'''
from ctsb.models.optimizers.core import Optimizer
from ctsb.models.optimizers.losses import mse

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
    def __init__(self, pred=None, loss=mse, learning_rate=0.01, hyperparameters={}):
        self.lr = learning_rate
        self.hyperparameters = hyperparameters
        if self._is_valid_pred(pred, raise_error=False):
            self.set_predict(pred, loss=loss)
        else:
            self.initialized = False


    def update(self, params, x, y, loss=None):
        """
        Description: Updates parameters based on correct value, loss and learning rate.
        Args:
            params (list/numpy.ndarray): Parameters of model pred method
            x (float): input to model
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
