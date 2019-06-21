# test the Random problem class

import ctsb
import ctsb.core
from ctsb.core import Problem
from ctsb.problems.simulated.random import Random
import jax.numpy as np
import matplotlib.pyplot as plt


def test_random():
    T = 100000
    problem = Random()
    assert problem.T == 0

    test_output = []
    for t in range(T):
        test_output.append(problem.step())

    assert problem.T == T
    plt.plot(test_output)
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    return


if __name__=="__main__":
    test_random()