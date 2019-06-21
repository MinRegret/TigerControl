# test the ARMA problem class

import ctsb
import ctsb.core
from ctsb.core import Problem
from ctsb.problems.simulated.arma import ARMA
import jax.numpy as np
import matplotlib.pyplot as plt


def test_arma():
    T = 100000
    p, q = 3, 3
    problem = ARMA(p, q)
    assert problem.T == 0

    test_output = []
    for t in range(T):
        test_output.append(problem.step())

    assert problem.T == T
    print(problem.phi)
    print(problem.psi)
    plt.plot(test_output)
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    return


if __name__=="__main__":
    test_arma()