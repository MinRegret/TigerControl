# test the Random problem class

import ctsb
from ctsb.problems.control.random import Random
import jax.numpy as np
import matplotlib.pyplot as plt


def test_random(steps=100, show_plot=False):
    T = steps
    problem = Random()
    problem.initialize()
    assert problem.T == 0

    test_output = []
    for t in range(T):
        test_output.append(problem.step())

    assert problem.T == T
    if show_plot:
        plt.plot(test_output)
        plt.title("random")
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    print("test_random passed")
    return


if __name__=="__main__":
    test_random()