"""
unit tests for GridSearch class
"""

import tigercontrol
from tigercontrol.utils.autotuning import GridSearch
from tigercontrol.utils.optimizers import *
import jax.numpy as np
import matplotlib.pyplot as plt
import itertools

def test_grid_search(show=False):
    test_grid_search_arma(show=show)
    print("test_grid_search passed")


def test_grid_search_arma(show=False):
    environment_id = "LDS-v0"
    controller_id = "GPC"
    environment_params = {'n':3, 'm':2}
    controller_params = {}
    loss = lambda a, b: np.sum((a-b)**2)
    search_space = {'optimizer':[]} # parameters for ARMA controller
    opts = [Adam, Adagrad, ONS, OGD]
    lr_start, lr_stop = 0, -4 # search learning rates from 10^start to 10^stop 
    learning_rates = np.logspace(lr_start, lr_stop, 1+2*np.abs(lr_start - lr_stop))
    for opt, lr in itertools.product(opts, learning_rates):
        search_space['optimizer'].append(opt(learning_rate=lr)) # create instance and append

    trials = 15
    hpo = GridSearch() # hyperparameter optimizer
    optimal_params, optimal_loss = hpo.search(controller_id, controller_params, environment_id, environment_params, loss, 
        search_space, trials=trials, smoothing=10, start_steps=100, verbose=show)

    if show:
        print("optimal loss: ", optimal_loss)
        print("optimal params: ", optimal_params)

    # test resulting controller params
    controller = tigercontrol.controllers(controller_id)
    controller.initialize(**optimal_params)
    environment = tigercontrol.environment(environment_id)
    x = environment.initialize(**environment_params)
    loss = []
    if show:
        print("run final test with optimal parameters")
    for t in range(5000):
        y_pred = controller.predict(x)
        y_true = environment.step()
        loss.append(mse(y_pred, y_true))
        controller.update(y_true)
        x = y_true

    if show:
        plt.plot(loss)
        plt.show(block=False)
        plt.pause(10)
        plt.close()



if __name__ == "__main__":
    test_grid_search(show=True)

