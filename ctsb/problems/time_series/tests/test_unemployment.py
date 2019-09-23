# test the crypto problem class

import tigercontrol
import jax.numpy as np
import matplotlib.pyplot as plt


def test_unemployment(steps=1000, show_plot=False, verbose=False):
    T = steps if steps < 800 else 800 # short time-series
    problem = tigercontrol.problem("Unemployment-v0")
    problem.initialize()
    assert problem.T == 0

    test_output = []
    for t in range(T):
        test_output.append(problem.step())

    assert problem.T == T
    
    info = problem.hidden()
    if verbose:
        print(info)
    if show_plot:
        plt.plot(test_output)
        plt.title("Unemployment")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    print("test_crypto passed")
    return


if __name__=="__main__":
    test_unemployment(show_plot=True)