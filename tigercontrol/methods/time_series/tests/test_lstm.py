# test the LSTM method class

import tigercontrol
import numpy as onp
import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
from tigercontrol.utils import generate_key
from tigercontrol.utils.download_tools import get_tigercontrol_dir
import os

# https://cnsviewer.corp.google.com/cns/jn-d/home/floods/hydro_method/datasets/processed/full/
def test_flood_FL(steps=61, show_plot=True):
    T = steps 
    n, m, l, d = 1, 1, 3, 10
    # problem = tigercontrol.problem("LDS-Control-v0")
    # y_true = problem.initialize(n, m, d)
    tigercontrol_dir = get_tigercontrol_dir()
    data_path = os.path.join(tigercontrol_dir, 'data/FL')
    arr_results = onp.load(data_path, encoding='bytes')
    method = tigercontrol.method("LSTM")
    method.initialize(n, m, l, d)
    loss = lambda pred, true: np.sum((pred - true)**2)
    num_batches = len(arr_results)
    # for i in range(num_batches):
    #    print("num_measurements:" + str(len(arr_results[i][2])))

    
    results = []
    print("num_batches: " + str(num_batches))
    for i in range(1000):
        print(i)
        for j in range(61):
            y_pred = method.predict(arr_results[i][2][j])
            y_true = arr_results[i][3][j]
            results.append(loss(y_true, y_pred))
            method.update(y_true)

    if show_plot:
        plt.plot(results)
        plt.title("LSTM method on LDS problem")
        plt.show(block=True)
        plt.close()
    print("test_lstm passed")
    return

def test_lstm(steps=100, show_plot=True):
    T = steps 
    n, m, l, d = 4, 5, 10, 10
    problem = tigercontrol.problem("LDS-Control-v0")
    y_true = problem.initialize(n, m, d)
    method = tigercontrol.method("LSTM")
    method.initialize(n, m, l, d)
    loss = lambda pred, true: np.sum((pred - true)**2)
 
    results = []
    for i in range(T):
        u = random.normal(generate_key(), (n,))
        y_pred = method.predict(u)
        y_true = problem.step(u)
        results.append(loss(y_true, y_pred))
        method.update(y_true)

    if show_plot:
        plt.plot(results)
        plt.title("LSTM method on LDS problem")
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    print("test_lstm passed")
    return

if __name__=="__main__":
    # test_lstm()
    test_flood_FL()