"""
unit tests for GridSearch class
"""

import tigercontrol
from tigercontrol.methods.autotuning import GridSearch
from tigercontrol.methods.optimizers import *
import jax.numpy as np
import matplotlib.pyplot as plt
import itertools

def test_grid_search(show=False):
    test_grid_search_arma(show=show)
    print("test_grid_search passed")


def test_grid_search_arma(show=False):
    problem_id = "ARMA-v0"
    method_id = "AutoRegressor"
    problem_params = {'p':3, 'q':2}
    method_params = {}
    loss = lambda a, b: np.sum((a-b)**2)
    search_space = {'p': [1,2,3,4,5], 'optimizer':[]} # parameters for ARMA method
    opts = [Adam, Adagrad, ONS, OGD]
    lr_start, lr_stop = 0, -4 # search learning rates from 10^start to 10^stop 
    learning_rates = np.logspace(lr_start, lr_stop, 1+2*np.abs(lr_start - lr_stop))
    for opt, lr in itertools.product(opts, learning_rates):
        search_space['optimizer'].append(opt(learning_rate=lr)) # create instance and append

    trials = 15
    hpo = GridSearch() # hyperparameter optimizer
    optimal_params, optimal_loss = hpo.search(method_id, method_params, problem_id, problem_params, loss, 
        search_space, trials=trials, smoothing=10, start_steps=100, verbose=show)

    if show:
        print("optimal loss: ", optimal_loss)
        print("optimal params: ", optimal_params)

    # test resulting method params
    method = tigercontrol.method(method_id)
    method.initialize(**optimal_params)
    problem = tigercontrol.problem(problem_id)
    x = problem.initialize(**problem_params)
    loss = []
    if show:
        print("run final test with optimal parameters")
    for t in range(5000):
        y_pred = method.predict(x)
        y_true = problem.step()
        loss.append(mse(y_pred, y_true))
        method.update(y_true)
        x = y_true

    if show:
        plt.plot(loss)
        plt.show(block=False)
        plt.pause(10)
        plt.close()



if __name__ == "__main__":
    test_grid_search(show=True)

