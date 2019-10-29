'''
Newton Step optimizer
'''

from tigerforecast.methods.optimizers.core import Optimizer
from tigerforecast.methods.optimizers.losses import mse
from tigerforecast import error
from jax import jit, grad
import jax.numpy as np

# regular numpy is necessary for cvxopt to work
import numpy as onp
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False


class ONS(Optimizer):
    """
    Online newton step algorithm.
    """

    def __init__(self, pred=None, loss=mse, learning_rate=1.0, hyperparameters={}):
        self.initialized = False
        self.lr = learning_rate
        self.hps = {'reg':0.00, 'eps':0.0001, 'max_norm':0, 'project':False, 'full_matrix':False}
        self.hps.update(hyperparameters)
        self.original_max_norm = self.hps['max_norm']
        for key, value in self.hps.items():
            if hasattr(self, key):
                raise error.InvalidInput("key {} is already an attribute in {}".format(key, self))
            setattr(self, key, value) # store all hyperparameters
        self.A, self.Ainv = None, None
        self.pred, self.loss = pred, loss
        self.numpyify = lambda m: onp.array(m).astype(onp.double) # maps jax.numpy to regular numpy

        if self._is_valid_pred(pred, raise_error=False) and self._is_valid_loss(loss, raise_error=False):
            self.set_predict(pred, loss=loss)

        @jit # partial update step for every matrix in method weights list
        def partial_update(A, Ainv, grad, w):
            A = A + np.outer(grad, grad)
            inv_grad = Ainv @ grad
            Ainv = Ainv - np.outer(inv_grad, inv_grad) / (1 + grad.T @ Ainv @ grad)
            new_grad = np.reshape(Ainv @ grad, w.shape)
            return A, Ainv, new_grad
        self.partial_update = partial_update

    def reset(self):
        self.A, self.Ainv = None, None
        if self.original_max_norm: self.max_norm = self.original_max_norm

    def norm_project(self, y, A, c):
        """ Project y using norm A on the convex set bounded by c. """
        if np.any(np.isnan(y)) or np.all(np.absolute(y) <= c):
            return y
        y_shape = y.shape
        y_reshaped = np.ravel(y)
        dim_y = y_reshaped.shape[0]
        P = matrix(self.numpyify(A))
        q = matrix(self.numpyify(-np.dot(A, y_reshaped)))
        G = matrix(self.numpyify(np.append(np.identity(dim_y), -np.identity(dim_y), axis=0)), tc='d')
        h = matrix(self.numpyify(np.repeat(c, 2 * dim_y)), tc='d')
        solution = np.array(onp.array(solvers.qp(P, q, G, h)['x'])).squeeze().reshape(y_shape)
        return solution

    def general_norm(self, x):
        x = np.asarray(x)
        if np.ndim(x) == 0:
            x = x[None]
        return np.linalg.norm(x)

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

        # get args
        grad = self.gradient(params, x, y, loss=loss) # defined in optimizers core class
        eta = self.lr
        
        # Make everything a list for generality
        is_list = True
        if(type(params) is not list):
            params = [params]
            grad = [grad]
            is_list = False

        # used to compute inverse matrix with respect to each parameter vector
        flat_grad = [np.ravel(dw) for dw in grad]

        # initialize A
        if self.A is None:
            self.A = [np.eye(dw.shape[0]) * self.eps for dw in flat_grad]
            self.Ainv = [np.eye(dw.shape[0]) * (1 / self.eps) for dw in flat_grad]

        # compute max norm and normalize accordingly
        if(self.max_norm):                     
            self.max_norm = np.maximum(self.max_norm, np.linalg.norm([self.general_norm(dw) for dw in flat_grad]))
            eta = eta * self.max_norm
            
        # partial_update automatically reshapes flat_grad into correct params shape
        new_values = [self.partial_update(A, Ainv, g, w) for (A, Ainv, g, w) in zip(self.A, self.Ainv, flat_grad, params)]
        self.A, self.Ainv, new_grad = list(map(list, zip(*new_values)))

        new_params = [w - eta * dw for (w, dw) in zip(params, new_grad)]

        if self.project:
            self.min_radius = np.maximum(self.min_radius, self.general_norm(y))
            norm = 5. * self.min_radius
            new_params = [self.norm_project(p, A, norm) for (p, A) in zip(new_params, self.A)]

        return new_params if is_list else new_params[0]

    def __str__(self):
        return "<ONS Optimizer, lr={}>".format(self.lr)


