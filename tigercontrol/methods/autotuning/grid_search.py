"""
Hyperparameter tuning using (optionally random) Grid Search.
"""

import tigercontrol
import jax
import jax.numpy as np
from jax import jit, grad

class GridSearch:
    """
    Description: Implements the equivalent of an AR(p) method - predicts a linear
    combination of the previous p observed values in a time-series
    """


    def __init__(self):
        pass

    def search(self, method_id, method_params, problem_id, problem_params, loss, search_space, trials=None, 
        smoothing=10, start_steps=100):
        """
        Description: Search for optimal method parameters
        Args:
            method_id (string): id of method
            method_params (dict): initial method parameters dict (updated by search space)
            problem_id (string): id of problem to try on
            problem_params (dict): problem parameters dict
            loss (function): a function mapping y_pred, y_true -> scalar loss
            search_space (dict): dict mapping parameter names to a finite set of options
            trials (int, None): number of random trials to sample from search space / try all parameters
            smoothing (int): loss computed over smoothing number of steps to decrease variance
            start_steps (int): minimum number of steps that the method gets to run for
        """
        self.method_id = method_id
        self.method_params = method_params
        self.problem_id = problem_id
        self.problem_params = problem_params
        self.loss = loss

        # helper method
        def _update_smoothing(l, val):
            """ update smoothing loss list with new val """
            return jax.ops.index_update(np.roll(l, 1), 0, val)
        self._update_smoothing = jax.jit(_update_smoothing)

        # store optimal params and optimal loss
        optimal_params, optimal_loss = {}, None
        while True:
            params = self._get_params(search_space, trials)
            curr_params = problem_params



    def _get_params(self, search_space, trials):
        return None


    def _run_test(self, method_params, smoothing, start_steps):
        """ Run a single test with given method params, using median stopping rule """
        # initialize problem and method
        method = tigercontrol.method(self.method_id)
        method.initialize(method_params)
        problem = tigercontrol.problem(self.problem_id)
        if problem.has_regressors:
            x, y_true = problem.initialize(self.problem_params)
        else:
            x = problem.initialize(self.problem_params)

        t = 0 # current time
        losses = [] # sorted losses, used to get median
        smooth_losses = np.zeros(smoothing) # store previous losses to get smooth loss
        while True: # run method until worse than median loss, ignoring first 100 steps
            t += 1
            y_pred = method.predict(x)
            if problem.has_regressors:
                method.update(y_true)
                loss = self.loss(y_pred, y_true)
            else:
                x = problem.step()
                method.update(x)
                loss = self.loss(y_pred, x)
            if t == 1: # fill all of smooth_losses with the first loss
                [_update_smoothing(smooth_losses, loss) for i in range(smoothing)]
            else: # else replace only the oldest loss
                _update_smoothing(smooth_losses, loss)
            smooth_loss = np.mean(smooth_loss)
            if t % smoothing == 0:
                _add_to_list(losses, smooth_loss)
                if _halting_rule(losses, smooth_loss) and t >= start_steps: break
        return smooth_loss

    def _add_to_list(self, l, val):
        """ add val to list l in sorted order """
        i = 0
        while i < len(l) and l[i] < val: i += 1
        l.insert(i, val)

    def _halting_rule(self, l, val, div=2): # div can be set to gamma > 2 to make stopping rule stricter
        """ return True if val is greater than median of list """
        if len(l) % 2 == 0:
            return val >= (l[int(len(l)/div)] + l[int(len(l)/div - 1)]) / 2
        return val >= l[int(len(l)/div)]


