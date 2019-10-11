# test the LDS problem class

import jax.numpy as np
import jax.random as random
import tigercontrol
import matplotlib.pyplot as plt
from tigercontrol.utils.random import generate_key


def test_lds(steps=1000, show_plot=False, verbose=False):
    T = steps
    n, m, d = 10, 3, 2 # state, input, and observation dimension
    problem = tigercontrol.problem("LDS-v0")
    problem.initialize(n, m, d, partially_observable=True, noise_distribution='normal')

    test_output = []
    for t in range(T):
        u = random.normal(generate_key(), shape=(m,))
        test_output.append(problem.step(u))

    info = problem.hidden()
    if verbose:
        print(info)

    assert problem.T == T
    if show_plot:
        plt.plot(test_output)
        plt.title("lds")
        plt.show(block=False)
        plt.pause(10)
        plt.close()
    print("test_lds_control passed")
    return


if __name__=="__main__":
    test_lds(show_plot=True)