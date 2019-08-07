'''
Newton Step optimizer
'''

from ctsb.models.optimizers.core import Optimizer
from ctsb.models.optimizers.losses import mse
from jax import jit, grad
import jax.numpy as np
import cvxopt

cvxopt.solvers.options['show_progress'] = False

class ONS(Optimizer):
    """
    Online newton step algorithm.
    """

    def __init__(self, pred=None, loss=mse, hyperparameters={}):
        
        self.initialized = False
        
        self.max_norm = 1.
        
        self.hyperparameters = {'beta':20., 'eps':0.1, 'fast_inverse':False, 'project':False, 'full_matrix':False}
        self.hyperparameters.update(hyperparameters)
        self.beta, self.eps = self.hyperparameters['beta'], self.hyperparameters['eps']
        self.fast_inverse, self.project = self.hyperparameters['fast_inverse'], self.hyperparameters['project']
        self.full_matrix = self.hyperparameters['full_matrix']

        self.A, self.Ainv = None, None

        self.pred, self.loss = pred, loss

        if self._is_valid_pred(pred, raise_error=False) and self._is_valid_loss(loss, raise_error=False):
            self.set_predict(pred, loss=loss)

    def norm_project(self, y, A, c):
        """ 
            Project y using norm A on the convex set bounded by c.
        """

        if (np.all(np.absolute(y) <= c)): # check if y already in K
            return y

        P = cvxopt.matrix(A)
        q = cvxopt.matrix(-A.dot(y))
        G = cvxopt.matrix(np.append(np.identity(len(y)), -np.identity(len(y)), axis=0), tc='d')
        h = cvxopt.matrix(np.repeat(c, 2 * len(y)), tc='d')

        return (np.array([1]) * cvxopt.solvers.qp(P, q, G, h)['x']).reshape(len(y),) # run quadratic program

    def general_norm(self, x):
        x = np.asarray(x)
        if np.ndim(x) == 0:
            x = x[None]
        return np.linalg.norm(x)

    def update(self, params, x, y, loss=mse):
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
        is_list = True

        # Make everything a list for generality
        if(type(params) is not list):
            params = [params]
            grad = [grad]
            is_list = False

        grad = [np.ravel(dw) for dw in grad]

        # initualize A
        if(self.A is None):
            self.A = [np.eye(dw.shape[0]) * self.eps for dw in grad]
            if(self.fast_inverse):
                self.Ainv = [np.eye(dw.shape[0]) * (1 / self.eps) for dw in grad]

        for i in range(len(grad)):
            self.A[i] += grad[i] @ grad[i].T # update
            # PROBLEM: grad[i] shape is 40x10!!!
            if(self.fast_inverse): 
                self.Ainv[i] -= (self.Ainv[i] @ grad[i]) @ (self.Ainv[i] @ grad[i]).T / \
                                (1 + grad[i].T @ self.Ainv[i] @ grad[i])

        # compute max norm for normalization                       
        self.max_norm = np.maximum(self.max_norm, np.linalg.norm([self.general_norm(dw) for dw in grad]))
        eta = (1 / self.max_norm) * (1. / self.beta)

        if(self.fast_inverse):
            new_params = [w - eta * np.reshape(Ainv @ dw, w.shape) \
                        for (w, Ainv, dw) in zip(params, self.Ainv, grad)]
        else:
            new_params = [w - eta * np.reshape(np.linalg.inv(A) @ dw, w.shape) \
                                for (w, A, dw) in zip(params, self.A, grad)]

        if(self.project):
            new_params = [self.norm_project(p, A, 2 * self.general_norm(y)) \
                    for (p, A) in zip(new_params, self.A)]
        
        if(not is_list):
            new_params = new_params[0]

        return new_params

