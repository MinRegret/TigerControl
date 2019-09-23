# test the LDS problem class

import jax.numpy as np
import jax.random as random
import tigercontrol
import matplotlib.pyplot as plt
from tigercontrol.utils.random import generate_key


def test_lds_control(steps=1000, show_plot=False, verbose=False):
    T = steps
    n, m, d = 5, 3, 10
    problem = tigercontrol.problem("LDS-Control-v0")
    problem.initialize(n, m, d)
    assert problem.T == 0

    test_output = []
    for t in range(T):
        u = random.normal(generate_key(),shape=(n,))
        test_output.append(problem.step(u))

    info = problem.hidden()
    if verbose:
        print(info)

    assert problem.T == T
    if show_plot:
        plt.plot(test_output)
        plt.title("lds")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    print("test_lds_control passed")
    return


if __name__=="__main__":
    test_lds_control(show_plot=True)