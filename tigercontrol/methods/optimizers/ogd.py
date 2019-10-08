'''
OGD optimizer
'''
import jax.numpy as np
from tigercontrol.methods.optimizers.core import Optimizer
from tigercontrol.methods.optimizers.losses import mse
from tigercontrol import error

class OGD(Optimizer):
    """
    Description: Ordinary Gradient Descent optimizer.
    Args:
        pred (function): a prediction function implemented with jax.numpy 
        loss (function): specifies loss function to be used; defaults to MSE
        learning_rate (float): learning rate
    Returns:
        None
    """
    def __init__(self, pred=None, loss=mse, learning_rate=1.0, hyperparameters={}):
        self.initialized = False
        self.lr = learning_rate
        self.hyperparameters = {'T':0, 'max_norm':True}
        self.hyperparameters.update(hyperparameters)
        for key, value in self.hyperparameters.items():
            if hasattr(self, key):
                raise error.InvalidInput("key {} is already an attribute in {}".format(key, self))
            setattr(self, key, value) # store all hyperparameters
        self.G = None
        self.pred = pred
        self.loss = loss
        if self._is_valid_pred(pred, raise_error=False) and self._is_valid_loss(loss, raise_error=False):
            self.set_predict(pred, loss=loss)

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

        self.T += 1
        grad = self.gradient(params, x, y, loss=loss) # defined in optimizers core class

        # Make everything a list for generality
        is_list = True
        if(type(params) is not list):
            params = [params]
            grad = [grad]
            is_list = False
    
        lr = self.lr / np.sqrt(self.T)
        if self.max_norm:
            self.max_norm = np.maximum(self.max_norm, np.linalg.norm([np.linalg.norm(dw) for dw in grad]))
            lr = self.lr / self.max_norm
        new_params = [w - lr * dw for (w, dw) in zip(params, grad)]

        return new_params if is_list else new_params[0]



