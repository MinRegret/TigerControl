# test the LSTM problem class

import jax.numpy as np
import jax.random as random
import tigercontrol
import matplotlib.pyplot as plt
from tigercontrol.utils.random import generate_key


def test_lstm_time_series(steps=1000, show_plot=False, verbose=False):
    T = steps
    n, m, d = 5, 3, 10
    problem = tigercontrol.problem("LSTM-TimeSeries-v0")
    problem.initialize(n, m, d)

    x_output = []
    y_output = []
    for t in range(T):
        x, y = problem.step()
        x_output.append(x)
        y_output.append(y)

    info = problem.hidden()
    if verbose:
        print(info)

    if show_plot:
        plt.plot(x_output)
        plt.plot(y_output)
        plt.title("lds")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    print("test_lstm_time_series passed")
    return


if __name__=="__main__":
    test_lstm_time_series(show_plot=True)