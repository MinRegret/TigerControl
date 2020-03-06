"""
Hyperparameter tuning using (optionally random) Grid Search.
""" 

import tigercontrol
from tigercontrol.utils.random import generate_key
import jax.numpy as np
import jax
from jax import jit, grad, random
import itertools


class GridSearch:
    """
    Description: Implements the equivalent of an AR(p) controller - predicts a linear
    combination of the previous p observed values in a time-series
    """
    def __init__(self):
        pass

    def search(self, controller_id, controller_params, environment_id, environment_params, loss, search_space, trials=None, 
        smoothing=10, min_steps=100, verbose=0):
        """
        Description: Search for optimal controller parameters
        Args:
            controller_id (string): id of controller
            controller_params (dict): initial controller parameters dict (updated by search space)
            environment_id (string): id of environment to try on
            environment_params (dict): environment parameters dict
            loss (function): a function mapping y_pred, y_true -> scalar loss
            search_space (dict): dict mapping parameter names to a finite set of options
            trials (int, None): number of random trials to sample from search space / try all parameters
            smoothing (int): loss computed over smoothing number of steps to decrease variance
            min_steps (int): minimum number of steps that the controller gets to run for
            verbose (int): if 1, print progress and current parameters
        """
        self.controller_id = controller_id
        self.controller_params = controller_params
        self.environment_id = environment_id
        self.environment_params = environment_params
        self.loss = loss

        # store the order to test parameters
        param_list = list(itertools.product(*[v for k, v in search_space.items()]))
        index = np.arange(len(param_list)) # np.random.shuffle doesn't work directly on non-JAX objects
        shuffled_index = random.shuffle(generate_key(), index)
        param_order = [param_list[i] for i in shuffled_index] # shuffle order of elements

        # helper controller
        def _update_smoothing(l, val):
            """ update smoothing loss list with new val """
            return jax.ops.index_update(np.roll(l, 1), 0, val)
        self._update_smoothing = jit(_update_smoothing)

        # store optimal params and optimal loss
        optimal_params, optimal_loss = {}, None
        t = 0
        for params in param_order: # loop over all params in the given order
            t += 1
            curr_params = controller_params.copy()
            curr_params.update({k:v for k, v in zip(search_space.keys(), params)})
            loss = self._run_test(curr_params, smoothing=smoothing, min_steps=min_steps, verbose=verbose)
            if not optimal_loss or loss < optimal_loss:
                optimal_params = curr_params
                optimal_loss = loss
            if t == trials: # break after trials number of attempts, unless trials is None
                break
        return optimal_params, optimal_loss


    def _run_test(self, controller_params, smoothing, min_steps, verbose=0):
        """ Run a single test with given controller params, using median stopping rule """
        # initialize environment and controller
        if verbose:
            print("Currently testing parameters: " + str(controller_params))
        controller = tigercontrol.controller(self.controller_id)
        controller.initialize(**controller_params)
        environment = tigercontrol.environment(self.environment_id)
        if environment.has_regressors:
            x, y_true = environment.reset(**self.environment_params)
        else:
            x = environment.reset(**self.environment_params)

        t = 0
        losses = [] # sorted losses, used to get median
        smooth_losses = np.zeros(smoothing) # store previous losses to get smooth loss
        while True: # run controller until worse than median loss, ignoring first 100 steps
            t += 1
            y_pred = controller.predict(x)
            if environment.has_regressors:
                controller.update(y_true)
                loss = self.loss(y_pred, y_true)
            else:
                x = environment.step()
                controller.update(x)
                loss = self.loss(y_pred, x)
            if t == 1: # fill all of smooth_losses with the first loss
                for i in range(smoothing):
                    smooth_losses = self._update_smoothing(smooth_losses, loss)
            else: # else replace only the oldest loss
                smooth_losses = self._update_smoothing(smooth_losses, loss)
            smooth_loss = np.mean(smooth_losses)
            if t % smoothing == 0:
                self._add_to_list(losses, smooth_loss)
                if self._halting_rule(losses, smooth_loss) and t >= min_steps: break
        if verbose:
            print("Final loss: ", smooth_loss)
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


