# test the LSTM problem class

import tigercontrol
from tigercontrol.utils.random import generate_key
import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt


def test_lstm_control(steps=1000, show_plot=False, verbose=False):
    T = steps
    n, m, h = 5, 3, 10
    problem = tigercontrol.problem("LSTM-Control-v0")
    problem.initialize(n, m, h)
    assert problem.T == 0

    test_output = []
    for t in range(T):
        if verbose and (t+1) * 10 % T == 0:
            print("{} timesteps".format(t+1))
        u = random.normal(generate_key(), shape=(n,))
        test_output.append(problem.step(u))

    info = problem.hidden()
    if verbose:
        print(info)
        
    assert problem.T == T
    if show_plot:
        plt.plot(test_output)
        plt.title("lstm")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    print("test_lstm_control passed")
    return


if __name__=="__main__":
    test_lstm_control()