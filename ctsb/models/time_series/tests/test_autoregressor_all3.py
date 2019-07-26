# test the Autogressor model class

import ctsb
import jax.numpy as np
import matplotlib.pyplot as plt
from ctsb.models.optimizers import *

def test_autoregressor(steps=100, show_plot=True):
    T = steps 
    p, q = 3, 0
    problem = ctsb.problem("ARMA-v0")
    x = problem.initialize(p, q)

    model_sgd = ctsb.model("AutoRegressor")
    model_sgd.initialize(p, optimizer=SGD)
    model_ogd = ctsb.model("AutoRegressor")
    model_ogd.initialize(p, optimizer=OGD)
    model_ada = ctsb.model("AutoRegressor")
    model_ada.initialize(p, optimizer=Adagrad)

    loss = lambda y_true, y_pred: np.sum((y_true - y_pred)**2)

    results_sgd = []
    results_ogd = []
    results_ada = []
    y_vals = []

    for i in range(T):
        y_pred_sgd = model_sgd.predict(x)
        y_pred_ogd = model_ogd.predict(x)
        y_pred_ada = model_ada.predict(x)
        
        y_true = problem.step()
        
        cur_loss_sgd = loss(y_true, y_pred_sgd)
        cur_loss_ogd = loss(y_true, y_pred_ogd)
        cur_loss_ada = loss(y_true, y_pred_ada)
        
        model_sgd.update(y_true)
        model_ogd.update(y_true)
        model_ada.update(y_true)
        
        x = y_true
        
        results_sgd.append(cur_loss_sgd)
        results_ogd.append(cur_loss_ogd)
        results_ada.append(cur_loss_ada)
        
        y_vals.append(y_true)

    if show_plot:
        plt.plot(results_sgd, label = "SGD")
        plt.plot(results_ogd, label = "OGD")
        plt.plot(results_ada, label = "Adagrad")
        plt.title("AutoRegressive model on ARMA problem")
        plt.legend()
        plt.show()
    print("test_autoregressor passed")
    return

if __name__=="__main__":
    test_autoregressor()